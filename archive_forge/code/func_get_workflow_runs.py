from __future__ import annotations
import collections
import urllib.parse
from base64 import b64encode
from collections.abc import Iterable
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Any
from deprecated import deprecated
import github.AdvisoryCredit
import github.AdvisoryVulnerability
import github.Artifact
import github.AuthenticatedUser
import github.Autolink
import github.Branch
import github.CheckRun
import github.CheckSuite
import github.Clones
import github.CodeScanAlert
import github.Commit
import github.CommitComment
import github.Comparison
import github.ContentFile
import github.DependabotAlert
import github.Deployment
import github.Download
import github.Environment
import github.EnvironmentDeploymentBranchPolicy
import github.EnvironmentProtectionRule
import github.EnvironmentProtectionRuleReviewer
import github.Event
import github.GitBlob
import github.GitCommit
import github.GithubObject
import github.GitRef
import github.GitRelease
import github.GitReleaseAsset
import github.GitTag
import github.GitTree
import github.Hook
import github.HookDelivery
import github.Invitation
import github.Issue
import github.IssueComment
import github.IssueEvent
import github.Label
import github.License
import github.Milestone
import github.NamedUser
import github.Notification
import github.Organization
import github.PaginatedList
import github.Path
import github.Permissions
import github.Project
import github.PublicKey
import github.PullRequest
import github.PullRequestComment
import github.Referrer
import github.RepositoryAdvisory
import github.RepositoryKey
import github.RepositoryPreferences
import github.Secret
import github.SelfHostedActionsRunner
import github.SourceImport
import github.Stargazer
import github.StatsCodeFrequency
import github.StatsCommitActivity
import github.StatsContributor
import github.StatsParticipation
import github.StatsPunchCard
import github.Tag
import github.Team
import github.Variable
import github.View
import github.Workflow
import github.WorkflowRun
from github import Consts
from github.Environment import Environment
from github.GithubObject import (
from github.PaginatedList import PaginatedList
def get_workflow_runs(self, actor: Opt[NamedUser]=NotSet, branch: Opt[Branch]=NotSet, event: Opt[str]=NotSet, status: Opt[str]=NotSet, exclude_pull_requests: Opt[bool]=NotSet, head_sha: Opt[str]=NotSet) -> PaginatedList[WorkflowRun]:
    """
        :calls: `GET /repos/{owner}/{repo}/actions/runs <https://docs.github.com/en/rest/reference/actions#list-workflow-runs-for-a-repository>`_
        :param actor: :class:`github.NamedUser.NamedUser` or string
        :param branch: :class:`github.Branch.Branch` or string
        :param event: string
        :param status: string `queued`, `in_progress`, `completed`, `success`, `failure`, `neutral`, `cancelled`, `skipped`, `timed_out`, or `action_required`
        :param exclude_pull_requests: bool
        :param head_sha: string

        :rtype: :class:`PaginatedList` of :class:`github.WorkflowRun.WorkflowRun`
        """
    assert is_optional(actor, (github.NamedUser.NamedUser, str)), actor
    assert is_optional(branch, (github.Branch.Branch, str)), branch
    assert is_optional(event, str), event
    assert is_optional(status, str), status
    assert is_optional(exclude_pull_requests, bool), exclude_pull_requests
    assert is_optional(head_sha, str), head_sha
    url_parameters: dict[str, Any] = {}
    if is_defined(actor):
        if isinstance(actor, github.NamedUser.NamedUser):
            url_parameters['actor'] = actor._identity
        else:
            url_parameters['actor'] = actor
    if is_defined(branch):
        if isinstance(branch, github.Branch.Branch):
            url_parameters['branch'] = branch.name
        else:
            url_parameters['branch'] = branch
    if is_defined(event):
        url_parameters['event'] = event
    if is_defined(status):
        url_parameters['status'] = status
    if is_defined(exclude_pull_requests) and exclude_pull_requests:
        url_parameters['exclude_pull_requests'] = 1
    if is_defined(head_sha):
        url_parameters['head_sha'] = head_sha
    return PaginatedList(github.WorkflowRun.WorkflowRun, self._requester, f'{self.url}/actions/runs', url_parameters, list_item='workflow_runs')