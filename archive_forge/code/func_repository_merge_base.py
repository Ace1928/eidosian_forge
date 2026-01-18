from typing import Any, Callable, Dict, Iterator, List, Optional, TYPE_CHECKING, Union
import requests
import gitlab
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import types, utils
@cli.register_custom_action('Project', ('refs',))
@exc.on_http_error(exc.GitlabGetError)
def repository_merge_base(self, refs: List[str], **kwargs: Any) -> Union[Dict[str, Any], requests.Response]:
    """Return a diff between two branches/commits.

        Args:
            refs: The refs to find the common ancestor of. Multiple refs can be passed.
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabGetError: If the server failed to perform the request

        Returns:
            The common ancestor commit (*not* a RESTObject)
        """
    path = f'/projects/{self.encoded_id}/repository/merge_base'
    query_data, _ = utils._transform_types(data={'refs': refs}, custom_types={'refs': types.ArrayAttribute}, transform_data=True)
    return self.manager.gitlab.http_get(path, query_data=query_data, **kwargs)