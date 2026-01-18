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
@cli.register_custom_action('ProjectIssue', ('move_after_id', 'move_before_id'))
@exc.on_http_error(exc.GitlabUpdateError)
def reorder(self, move_after_id: Optional[int]=None, move_before_id: Optional[int]=None, **kwargs: Any) -> None:
    """Reorder an issue on a board.

        Args:
            move_after_id: ID of an issue that should be placed after this issue
            move_before_id: ID of an issue that should be placed before this issue
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabUpdateError: If the issue could not be reordered
        """
    path = f'{self.manager.path}/{self.encoded_id}/reorder'
    data: Dict[str, Any] = {}
    if move_after_id is not None:
        data['move_after_id'] = move_after_id
    if move_before_id is not None:
        data['move_before_id'] = move_before_id
    server_data = self.manager.gitlab.http_put(path, post_data=data, **kwargs)
    if TYPE_CHECKING:
        assert isinstance(server_data, dict)
    self._update_attrs(server_data)