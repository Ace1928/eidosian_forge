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
def create_milestone(self, title: str, state: Opt[str]=NotSet, description: Opt[str]=NotSet, due_on: Opt[date]=NotSet) -> Milestone:
    """
        :calls: `POST /repos/{owner}/{repo}/milestones <https://docs.github.com/en/rest/reference/issues#milestones>`_
        :param title: string
        :param state: string
        :param description: string
        :param due_on: datetime
        :rtype: :class:`github.Milestone.Milestone`
        """
    assert isinstance(title, str), title
    assert is_optional(state, str), state
    assert is_optional(description, str), description
    assert is_optional(due_on, (datetime, date)), due_on
    post_parameters = {'title': title}
    if is_defined(state):
        post_parameters['state'] = state
    if is_defined(description):
        post_parameters['description'] = description
    if is_defined(due_on):
        if isinstance(due_on, date):
            post_parameters['due_on'] = due_on.strftime('%Y-%m-%dT%H:%M:%SZ')
        else:
            post_parameters['due_on'] = due_on.isoformat()
    headers, data = self._requester.requestJsonAndCheck('POST', f'{self.url}/milestones', input=post_parameters)
    return github.Milestone.Milestone(self._requester, headers, data, completed=True)