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
def create_deployment(self, ref: str, task: Opt[str]=NotSet, auto_merge: Opt[bool]=NotSet, required_contexts: Opt[list[str]]=NotSet, payload: Opt[dict[str, Any]]=NotSet, environment: Opt[str]=NotSet, description: Opt[str]=NotSet, transient_environment: Opt[bool]=NotSet, production_environment: Opt[bool]=NotSet) -> Deployment:
    """
        :calls: `POST /repos/{owner}/{repo}/deployments <https://docs.github.com/en/rest/reference/repos#deployments>`_
        :param: ref: string
        :param: task: string
        :param: auto_merge: bool
        :param: required_contexts: list of status contexts
        :param: payload: dict
        :param: environment: string
        :param: description: string
        :param: transient_environment: bool
        :param: production_environment: bool
        :rtype: :class:`github.Deployment.Deployment`
        """
    assert isinstance(ref, str), ref
    assert is_optional(task, str), task
    assert is_optional(auto_merge, bool), auto_merge
    assert is_optional(required_contexts, list), required_contexts
    assert is_optional(payload, dict), payload
    assert is_optional(environment, str), environment
    assert is_optional(description, str), description
    assert is_optional(transient_environment, bool), transient_environment
    assert is_optional(production_environment, bool), production_environment
    post_parameters: dict[str, Any] = {'ref': ref}
    if is_defined(task):
        post_parameters['task'] = task
    if is_defined(auto_merge):
        post_parameters['auto_merge'] = auto_merge
    if is_defined(required_contexts):
        post_parameters['required_contexts'] = required_contexts
    if is_defined(payload):
        post_parameters['payload'] = payload
    if is_defined(environment):
        post_parameters['environment'] = environment
    if is_defined(description):
        post_parameters['description'] = description
    if is_defined(transient_environment):
        post_parameters['transient_environment'] = transient_environment
    if is_defined(production_environment):
        post_parameters['production_environment'] = production_environment
    headers, data = self._requester.requestJsonAndCheck('POST', f'{self.url}/deployments', input=post_parameters, headers={'Accept': Consts.deploymentEnhancementsPreview})
    return github.Deployment.Deployment(self._requester, headers, data, completed=True)