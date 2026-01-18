from typing import Any, cast, Dict, Optional, Tuple, TYPE_CHECKING, Union
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import types
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
from .award_emojis import ProjectIssueAwardEmojiManager  # noqa: F401
from .discussions import ProjectIssueDiscussionManager  # noqa: F401
from .events import (  # noqa: F401
from .notes import ProjectIssueNoteManager  # noqa: F401
@cli.register_custom_action('ProjectIssue')
@exc.on_http_error(exc.GitlabGetError)
def related_merge_requests(self, **kwargs: Any) -> Dict[str, Any]:
    """List merge requests related to the issue.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabGetErrot: If the merge requests could not be retrieved

        Returns:
            The list of merge requests.
        """
    path = f'{self.manager.path}/{self.encoded_id}/related_merge_requests'
    result = self.manager.gitlab.http_get(path, **kwargs)
    if TYPE_CHECKING:
        assert isinstance(result, dict)
    return result