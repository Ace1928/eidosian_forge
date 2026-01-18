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
def create_issue(self, title: str, body: Opt[str]=NotSet, assignee: NamedUser | Opt[str]=NotSet, milestone: Opt[Milestone]=NotSet, labels: list[Label] | Opt[list[str]]=NotSet, assignees: Opt[list[str]] | list[NamedUser]=NotSet) -> Issue:
    """
        :calls: `POST /repos/{owner}/{repo}/issues <https://docs.github.com/en/rest/reference/issues>`_
        :param title: string
        :param body: string
        :param assignee: string or :class:`github.NamedUser.NamedUser`
        :param assignees: list of string or :class:`github.NamedUser.NamedUser`
        :param milestone: :class:`github.Milestone.Milestone`
        :param labels: list of :class:`github.Label.Label`
        :rtype: :class:`github.Issue.Issue`
        """
    assert isinstance(title, str), title
    assert is_optional(body, str), body
    assert is_optional(assignee, (str, github.NamedUser.NamedUser)), assignee
    assert is_optional_list(assignees, (github.NamedUser.NamedUser, str)), assignees
    assert is_optional(milestone, github.Milestone.Milestone), milestone
    assert is_optional_list(labels, (github.Label.Label, str)), labels
    post_parameters: dict[str, Any] = {'title': title}
    if is_defined(body):
        post_parameters['body'] = body
    if is_defined(assignee):
        if isinstance(assignee, github.NamedUser.NamedUser):
            post_parameters['assignee'] = assignee._identity
        else:
            post_parameters['assignee'] = assignee
    if is_defined(assignees):
        post_parameters['assignees'] = [element._identity if isinstance(element, github.NamedUser.NamedUser) else element for element in assignees]
    if is_defined(milestone):
        post_parameters['milestone'] = milestone._identity
    if is_defined(labels):
        post_parameters['labels'] = [element.name if isinstance(element, github.Label.Label) else element for element in labels]
    headers, data = self._requester.requestJsonAndCheck('POST', f'{self.url}/issues', input=post_parameters)
    return github.Issue.Issue(self._requester, headers, data, completed=True)