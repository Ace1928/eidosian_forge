from typing import Any, Callable, Dict, Iterator, List, Optional, TYPE_CHECKING, Union
import requests
import gitlab
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import types, utils
@cli.register_custom_action('Project')
@exc.on_http_error(exc.GitlabDeleteError)
def delete_merged_branches(self, **kwargs: Any) -> None:
    """Delete merged branches.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabDeleteError: If the server failed to perform the request
        """
    path = f'/projects/{self.encoded_id}/repository/merged_branches'
    self.manager.gitlab.http_delete(path, **kwargs)