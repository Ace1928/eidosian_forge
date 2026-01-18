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
def create_git_release(self, tag: str, name: str, message: str, draft: bool=False, prerelease: bool=False, generate_release_notes: bool=False, target_commitish: Opt[str]=NotSet) -> GitRelease:
    """
        :calls: `POST /repos/{owner}/{repo}/releases <https://docs.github.com/en/rest/reference/repos#releases>`_
        :param tag: string
        :param name: string
        :param message: string
        :param draft: bool
        :param prerelease: bool
        :param generate_release_notes: bool
        :param target_commitish: string or :class:`github.Branch.Branch` or :class:`github.Commit.Commit` or :class:`github.GitCommit.GitCommit`
        :rtype: :class:`github.GitRelease.GitRelease`
        """
    assert isinstance(tag, str), tag
    assert isinstance(name, str), name
    assert isinstance(message, str), message
    assert isinstance(draft, bool), draft
    assert isinstance(prerelease, bool), prerelease
    assert isinstance(generate_release_notes, bool), generate_release_notes
    assert is_optional(target_commitish, (str, github.Branch.Branch, github.Commit.Commit, github.GitCommit.GitCommit)), target_commitish
    post_parameters = {'tag_name': tag, 'name': name, 'body': message, 'draft': draft, 'prerelease': prerelease, 'generate_release_notes': generate_release_notes}
    if isinstance(target_commitish, str):
        post_parameters['target_commitish'] = target_commitish
    elif isinstance(target_commitish, github.Branch.Branch):
        post_parameters['target_commitish'] = target_commitish.name
    elif isinstance(target_commitish, (github.Commit.Commit, github.GitCommit.GitCommit)):
        post_parameters['target_commitish'] = target_commitish.sha
    headers, data = self._requester.requestJsonAndCheck('POST', f'{self.url}/releases', input=post_parameters)
    return github.GitRelease.GitRelease(self._requester, headers, data, completed=True)