from typing import (
import requests
from gitlab import cli, client
from gitlab import exceptions as exc
from gitlab import types, utils
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
from .access_requests import ProjectAccessRequestManager  # noqa: F401
from .artifacts import ProjectArtifactManager  # noqa: F401
from .audit_events import ProjectAuditEventManager  # noqa: F401
from .badges import ProjectBadgeManager  # noqa: F401
from .boards import ProjectBoardManager  # noqa: F401
from .branches import ProjectBranchManager, ProjectProtectedBranchManager  # noqa: F401
from .ci_lint import ProjectCiLintManager  # noqa: F401
from .clusters import ProjectClusterManager  # noqa: F401
from .commits import ProjectCommitManager  # noqa: F401
from .container_registry import ProjectRegistryRepositoryManager  # noqa: F401
from .custom_attributes import ProjectCustomAttributeManager  # noqa: F401
from .deploy_keys import ProjectKeyManager  # noqa: F401
from .deploy_tokens import ProjectDeployTokenManager  # noqa: F401
from .deployments import ProjectDeploymentManager  # noqa: F401
from .environments import (  # noqa: F401
from .events import ProjectEventManager  # noqa: F401
from .export_import import ProjectExportManager, ProjectImportManager  # noqa: F401
from .files import ProjectFileManager  # noqa: F401
from .hooks import ProjectHookManager  # noqa: F401
from .integrations import ProjectIntegrationManager, ProjectServiceManager  # noqa: F401
from .invitations import ProjectInvitationManager  # noqa: F401
from .issues import ProjectIssueManager  # noqa: F401
from .iterations import ProjectIterationManager  # noqa: F401
from .job_token_scope import ProjectJobTokenScopeManager  # noqa: F401
from .jobs import ProjectJobManager  # noqa: F401
from .labels import ProjectLabelManager  # noqa: F401
from .members import ProjectMemberAllManager, ProjectMemberManager  # noqa: F401
from .merge_request_approvals import (  # noqa: F401
from .merge_requests import ProjectMergeRequestManager  # noqa: F401
from .merge_trains import ProjectMergeTrainManager  # noqa: F401
from .milestones import ProjectMilestoneManager  # noqa: F401
from .notes import ProjectNoteManager  # noqa: F401
from .notification_settings import ProjectNotificationSettingsManager  # noqa: F401
from .packages import GenericPackageManager, ProjectPackageManager  # noqa: F401
from .pages import ProjectPagesDomainManager  # noqa: F401
from .pipelines import (  # noqa: F401
from .project_access_tokens import ProjectAccessTokenManager  # noqa: F401
from .push_rules import ProjectPushRulesManager  # noqa: F401
from .releases import ProjectReleaseManager  # noqa: F401
from .repositories import RepositoryMixin
from .resource_groups import ProjectResourceGroupManager
from .runners import ProjectRunnerManager  # noqa: F401
from .secure_files import ProjectSecureFileManager  # noqa: F401
from .snippets import ProjectSnippetManager  # noqa: F401
from .statistics import (  # noqa: F401
from .tags import ProjectProtectedTagManager, ProjectTagManager  # noqa: F401
from .triggers import ProjectTriggerManager  # noqa: F401
from .users import ProjectUserManager  # noqa: F401
from .variables import ProjectVariableManager  # noqa: F401
from .wikis import ProjectWikiManager  # noqa: F401
@exc.on_http_error(exc.GitlabImportError)
def remote_import(self, url: str, path: str, name: Optional[str]=None, namespace: Optional[str]=None, overwrite: bool=False, override_params: Optional[Dict[str, Any]]=None, **kwargs: Any) -> Union[Dict[str, Any], requests.Response]:
    """Import a project from an archive file stored on a remote URL.

        Args:
            url: URL for the file containing the project data to import
            path: Name and path for the new project
            name: The name of the project to import. If not provided,
                defaults to the path of the project.
            namespace: The ID or path of the namespace that the project
                will be imported to
            overwrite: If True overwrite an existing project with the
                same path
            override_params: Set the specific settings for the project
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabImportError: If the server failed to perform the request

        Returns:
            A representation of the import status.
        """
    data = {'path': path, 'overwrite': str(overwrite), 'url': url}
    if override_params:
        for k, v in override_params.items():
            data[f'override_params[{k}]'] = v
    if name is not None:
        data['name'] = name
    if namespace:
        data['namespace'] = namespace
    return self.gitlab.http_post('/projects/remote-import', post_data=data, **kwargs)