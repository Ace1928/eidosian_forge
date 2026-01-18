from typing import Any, BinaryIO, cast, Dict, List, Optional, Type, TYPE_CHECKING, Union
import requests
import gitlab
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import types
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
from .access_requests import GroupAccessRequestManager  # noqa: F401
from .audit_events import GroupAuditEventManager  # noqa: F401
from .badges import GroupBadgeManager  # noqa: F401
from .boards import GroupBoardManager  # noqa: F401
from .clusters import GroupClusterManager  # noqa: F401
from .container_registry import GroupRegistryRepositoryManager  # noqa: F401
from .custom_attributes import GroupCustomAttributeManager  # noqa: F401
from .deploy_tokens import GroupDeployTokenManager  # noqa: F401
from .epics import GroupEpicManager  # noqa: F401
from .export_import import GroupExportManager, GroupImportManager  # noqa: F401
from .group_access_tokens import GroupAccessTokenManager  # noqa: F401
from .hooks import GroupHookManager  # noqa: F401
from .invitations import GroupInvitationManager  # noqa: F401
from .issues import GroupIssueManager  # noqa: F401
from .iterations import GroupIterationManager  # noqa: F401
from .labels import GroupLabelManager  # noqa: F401
from .members import (  # noqa: F401
from .merge_requests import GroupMergeRequestManager  # noqa: F401
from .milestones import GroupMilestoneManager  # noqa: F401
from .notification_settings import GroupNotificationSettingsManager  # noqa: F401
from .packages import GroupPackageManager  # noqa: F401
from .projects import GroupProjectManager, SharedProjectManager  # noqa: F401
from .push_rules import GroupPushRulesManager
from .runners import GroupRunnerManager  # noqa: F401
from .statistics import GroupIssuesStatisticsManager  # noqa: F401
from .variables import GroupVariableManager  # noqa: F401
from .wikis import GroupWikiManager  # noqa: F401
@exc.on_http_error(exc.GitlabImportError)
def import_group(self, file: BinaryIO, path: str, name: str, parent_id: Optional[Union[int, str]]=None, **kwargs: Any) -> Union[Dict[str, Any], requests.Response]:
    """Import a group from an archive file.

        Args:
            file: Data or file object containing the group
            path: The path for the new group to be imported.
            name: The name for the new group.
            parent_id: ID of a parent group that the group will
                be imported into.
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabImportError: If the server failed to perform the request

        Returns:
            A representation of the import status.
        """
    files = {'file': ('file.tar.gz', file, 'application/octet-stream')}
    data: Dict[str, Any] = {'path': path, 'name': name}
    if parent_id is not None:
        data['parent_id'] = parent_id
    return self.gitlab.http_post('/groups/import', post_data=data, files=files, **kwargs)