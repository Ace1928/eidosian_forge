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
def create_source_import(self, vcs: str, vcs_url: str, vcs_username: Opt[str]=NotSet, vcs_password: Opt[str]=NotSet) -> SourceImport:
    """
        :calls: `PUT /repos/{owner}/{repo}/import <https://docs.github.com/en/rest/reference/migrations#start-an-import>`_
        :param vcs: string
        :param vcs_url: string
        :param vcs_username: string
        :param vcs_password: string
        :rtype: :class:`github.SourceImport.SourceImport`
        """
    assert isinstance(vcs, str), vcs
    assert isinstance(vcs_url, str), vcs_url
    assert is_optional(vcs_username, str), vcs_username
    assert is_optional(vcs_password, str), vcs_password
    put_parameters = {'vcs': vcs, 'vcs_url': vcs_url}
    if is_defined(vcs_username):
        put_parameters['vcs_username'] = vcs_username
    if is_defined(vcs_password):
        put_parameters['vcs_password'] = vcs_password
    import_header = {'Accept': Consts.mediaTypeImportPreview}
    headers, data = self._requester.requestJsonAndCheck('PUT', f'{self.url}/import', headers=import_header, input=put_parameters)
    return github.SourceImport.SourceImport(self._requester, headers, data, completed=False)