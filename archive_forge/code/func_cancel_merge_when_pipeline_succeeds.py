from typing import Any, cast, Dict, Optional, TYPE_CHECKING, Union
import requests
import gitlab
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import types
from gitlab.base import RESTManager, RESTObject, RESTObjectList
from gitlab.mixins import (
from gitlab.types import RequiredOptional
from .award_emojis import ProjectMergeRequestAwardEmojiManager  # noqa: F401
from .commits import ProjectCommit, ProjectCommitManager
from .discussions import ProjectMergeRequestDiscussionManager  # noqa: F401
from .draft_notes import ProjectMergeRequestDraftNoteManager
from .events import (  # noqa: F401
from .issues import ProjectIssue, ProjectIssueManager
from .merge_request_approvals import (  # noqa: F401
from .notes import ProjectMergeRequestNoteManager  # noqa: F401
from .pipelines import ProjectMergeRequestPipelineManager  # noqa: F401
from .reviewers import ProjectMergeRequestReviewerDetailManager
@cli.register_custom_action('ProjectMergeRequest')
@exc.on_http_error(exc.GitlabMROnBuildSuccessError)
def cancel_merge_when_pipeline_succeeds(self, **kwargs: Any) -> Dict[str, str]:
    """Cancel merge when the pipeline succeeds.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabMROnBuildSuccessError: If the server could not handle the
                request

        Returns:
            dict of the parsed json returned by the server
        """
    path = f'{self.manager.path}/{self.encoded_id}/cancel_merge_when_pipeline_succeeds'
    server_data = self.manager.gitlab.http_post(path, **kwargs)
    if TYPE_CHECKING:
        assert isinstance(server_data, dict)
    return server_data