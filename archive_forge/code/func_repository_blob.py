from typing import Any, Callable, Dict, Iterator, List, Optional, TYPE_CHECKING, Union
import requests
import gitlab
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import types, utils
@cli.register_custom_action('Project', ('sha',))
@exc.on_http_error(exc.GitlabGetError)
def repository_blob(self, sha: str, **kwargs: Any) -> Union[Dict[str, Any], requests.Response]:
    """Return a file by blob SHA.

        Args:
            sha: ID of the blob
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabGetError: If the server failed to perform the request

        Returns:
            The blob content and metadata
        """
    path = f'/projects/{self.encoded_id}/repository/blobs/{sha}'
    return self.manager.gitlab.http_get(path, **kwargs)