import enum
from types import ModuleType
from typing import (
import requests
import gitlab
from gitlab import base, cli
from gitlab import exceptions as exc
from gitlab import utils
@cli.register_custom_action(('ProjectIssue', 'ProjectMergeRequest'))
@exc.on_http_error(exc.GitlabTimeTrackingError)
def reset_spent_time(self, **kwargs: Any) -> Dict[str, Any]:
    """Resets the time spent working on the object.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabTimeTrackingError: If the time tracking update cannot be done
        """
    path = f'{self.manager.path}/{self.encoded_id}/reset_spent_time'
    result = self.manager.gitlab.http_post(path, **kwargs)
    if TYPE_CHECKING:
        assert not isinstance(result, requests.Response)
    return result