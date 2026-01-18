from typing import Any, Callable, Dict, Iterator, List, Optional, TYPE_CHECKING, Union
import requests
import gitlab
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import types, utils
@cli.register_custom_action('Project')
@exc.on_http_error(exc.GitlabGetError)
def repository_contributors(self, **kwargs: Any) -> Union[gitlab.client.GitlabList, List[Dict[str, Any]]]:
    """Return a list of contributors for the project.

        Args:
            all: If True, return all the items, without pagination
            per_page: Number of items to retrieve per request
            page: ID of the page to return (starts with page 1)
            iterator: If set to True and no pagination option is
                defined, return a generator instead of a list
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabGetError: If the server failed to perform the request

        Returns:
            The contributors
        """
    path = f'/projects/{self.encoded_id}/repository/contributors'
    return self.manager.gitlab.http_list(path, **kwargs)