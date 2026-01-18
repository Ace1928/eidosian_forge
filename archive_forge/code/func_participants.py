import enum
from types import ModuleType
from typing import (
import requests
import gitlab
from gitlab import base, cli
from gitlab import exceptions as exc
from gitlab import utils
@cli.register_custom_action(('ProjectMergeRequest', 'ProjectIssue'))
@exc.on_http_error(exc.GitlabListError)
def participants(self, **kwargs: Any) -> Dict[str, Any]:
    """List the participants.

        Args:
            all: If True, return all the items, without pagination
            per_page: Number of items to retrieve per request
            page: ID of the page to return (starts with page 1)
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabListError: If the list could not be retrieved

        Returns:
            The list of participants
        """
    path = f'{self.manager.path}/{self.encoded_id}/participants'
    result = self.manager.gitlab.http_get(path, **kwargs)
    if TYPE_CHECKING:
        assert not isinstance(result, requests.Response)
    return result