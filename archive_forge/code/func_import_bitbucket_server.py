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
def import_bitbucket_server(self, bitbucket_server_url: str, bitbucket_server_username: str, personal_access_token: str, bitbucket_server_project: str, bitbucket_server_repo: str, new_name: Optional[str]=None, target_namespace: Optional[str]=None, **kwargs: Any) -> Union[Dict[str, Any], requests.Response]:
    """Import a project from BitBucket Server to Gitlab (schedule the import)

        This method will return when an import operation has been safely queued,
        or an error has occurred. After triggering an import, check the
        ``import_status`` of the newly created project to detect when the import
        operation has completed.

        .. note::
            This request may take longer than most other API requests.
            So this method will specify a 60 second default timeout if none is
            specified.
            A timeout can be specified via kwargs to override this functionality.

        Args:
            bitbucket_server_url: Bitbucket Server URL
            bitbucket_server_username: Bitbucket Server Username
            personal_access_token: Bitbucket Server personal access
                token/password
            bitbucket_server_project: Bitbucket Project Key
            bitbucket_server_repo: Bitbucket Repository Name
            new_name: New repository name (Optional)
            target_namespace: Namespace to import repository into.
                Supports subgroups like /namespace/subgroup (Optional)
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabListError: If the server failed to perform the request

        Returns:
            A representation of the import status.

        Example:

        .. code-block:: python

            gl = gitlab.Gitlab_from_config()
            print("Triggering import")
            result = gl.projects.import_bitbucket_server(
                bitbucket_server_url="https://some.server.url",
                bitbucket_server_username="some_bitbucket_user",
                personal_access_token="my_password_or_access_token",
                bitbucket_server_project="my_project",
                bitbucket_server_repo="my_repo",
                new_name="gl_project_name",
                target_namespace="gl_project_path"
            )
            project = gl.projects.get(ret['id'])
            print("Waiting for import to complete")
            while project.import_status == u'started':
                time.sleep(1.0)
                project = gl.projects.get(project.id)
            print("BitBucket import complete")

        """
    data = {'bitbucket_server_url': bitbucket_server_url, 'bitbucket_server_username': bitbucket_server_username, 'personal_access_token': personal_access_token, 'bitbucket_server_project': bitbucket_server_project, 'bitbucket_server_repo': bitbucket_server_repo}
    if new_name:
        data['new_name'] = new_name
    if target_namespace:
        data['target_namespace'] = target_namespace
    if 'timeout' not in kwargs or self.gitlab.timeout is None or self.gitlab.timeout < 60.0:
        kwargs['timeout'] = 60.0
    result = self.gitlab.http_post('/import/bitbucket_server', post_data=data, **kwargs)
    return result