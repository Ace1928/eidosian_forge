import os
import re
import time
from typing import (
from urllib import parse
import requests
import gitlab
import gitlab.config
import gitlab.const
import gitlab.exceptions
from gitlab import _backends, utils
def http_list(self, path: str, query_data: Optional[Dict[str, Any]]=None, *, iterator: Optional[bool]=None, **kwargs: Any) -> Union['GitlabList', List[Dict[str, Any]]]:
    """Make a GET request to the Gitlab server for list-oriented queries.

        Args:
            path: Path or full URL to query ('/projects' or
                        'http://whatever/v4/api/projects')
            query_data: Data to send as query parameters
            iterator: Indicate if should return a generator (True)
            **kwargs: Extra options to send to the server (e.g. sudo, page,
                      per_page)

        Returns:
            A list of the objects returned by the server. If `iterator` is
            True and no pagination-related arguments (`page`, `per_page`,
            `get_all`) are defined then a GitlabList object (generator) is returned
            instead. This object will make API calls when needed to fetch the
            next items from the server.

        Raises:
            GitlabHttpError: When the return code is not 2xx
            GitlabParsingError: If the json data could not be parsed
        """
    query_data = query_data or {}
    get_all = kwargs.pop('get_all', None)
    if get_all is None:
        get_all = kwargs.pop('all', None)
    url = self._build_url(path)
    page = kwargs.get('page')
    if iterator and page is not None:
        arg_used_message = f'iterator={iterator}'
        utils.warn(message=f'`{arg_used_message}` and `page={page}` were both specified. `{arg_used_message}` will be ignored and a `list` will be returned.', category=UserWarning)
    if iterator and page is None:
        return GitlabList(self, url, query_data, **kwargs)
    if get_all is True:
        return list(GitlabList(self, url, query_data, **kwargs))
    gl_list = GitlabList(self, url, query_data, get_next=False, **kwargs)
    items = list(gl_list)

    def should_emit_warning() -> bool:
        if get_all is False:
            return False
        if page is not None:
            return False
        if gl_list.per_page is None:
            return False
        if len(items) < gl_list.per_page:
            return False
        if gl_list.total is not None and len(items) >= gl_list.total:
            return False
        return True
    if not should_emit_warning():
        return items
    total_items = 'many' if gl_list.total is None else gl_list.total
    utils.warn(message=f'Calling a `list()` method without specifying `get_all=True` or `iterator=True` will return a maximum of {gl_list.per_page} items. Your query returned {len(items)} of {total_items} items. See {_PAGINATION_URL} for more details. If this was done intentionally, then this warning can be supressed by adding the argument `get_all=False` to the `list()` call.', category=UserWarning)
    return items