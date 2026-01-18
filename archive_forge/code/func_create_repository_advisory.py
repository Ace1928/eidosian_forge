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
def create_repository_advisory(self, summary: str, description: str, severity_or_cvss_vector_string: str, cve_id: str | None=None, vulnerabilities: Iterable[github.AdvisoryVulnerability.AdvisoryVulnerabilityInput] | None=None, cwe_ids: Iterable[str] | None=None, credits: Iterable[github.AdvisoryCredit.AdvisoryCredit] | None=None) -> github.RepositoryAdvisory.RepositoryAdvisory:
    """
        :calls: `POST /repos/{owner}/{repo}/security-advisories <https://docs.github.com/en/rest/security-advisories/repository-advisories>`_
        :param summary: string
        :param description: string
        :param severity_or_cvss_vector_string: string
        :param cve_id: string
        :param vulnerabilities: iterable of :class:`github.AdvisoryVulnerability.AdvisoryVulnerabilityInput`
        :param cwe_ids: iterable of string
        :param credits: iterable of :class:`github.AdvisoryCredit.AdvisoryCredit`
        :rtype: :class:`github.RepositoryAdvisory.RepositoryAdvisory`
        """
    return self.__create_repository_advisory(summary=summary, description=description, severity_or_cvss_vector_string=severity_or_cvss_vector_string, cve_id=cve_id, vulnerabilities=vulnerabilities, cwe_ids=cwe_ids, credits=credits, private_vulnerability_reporting=False)