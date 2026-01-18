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
def get_views_traffic(self, per: Opt[str]=NotSet) -> None | dict[str, int | list[View]]:
    """
        :calls: `GET /repos/{owner}/{repo}/traffic/views <https://docs.github.com/en/rest/reference/repos#traffic>`_
        :param per: string, must be one of day or week, day by default
        """
    assert per in ['day', 'week', NotSet], 'per must be day or week, day by default'
    url_parameters = dict()
    if is_defined(per):
        url_parameters['per'] = per
    headers, data = self._requester.requestJsonAndCheck('GET', f'{self.url}/traffic/views', parameters=url_parameters)
    if isinstance(data, dict) and 'views' in data and isinstance(data['views'], list):
        data['views'] = [github.View.View(self._requester, headers, item, completed=True) for item in data['views']]
        return data