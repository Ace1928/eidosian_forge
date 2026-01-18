from __future__ import annotations
import pickle
import urllib.parse
import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Any, BinaryIO, TypeVar
import urllib3
from urllib3.util import Retry
import github.ApplicationOAuth
import github.Auth
import github.AuthenticatedUser
import github.Enterprise
import github.Event
import github.Gist
import github.GithubApp
import github.GithubIntegration
import github.GithubRetry
import github.GitignoreTemplate
import github.GlobalAdvisory
import github.License
import github.NamedUser
import github.Topic
from github import Consts
from github.GithubIntegration import GithubIntegration
from github.GithubObject import GithubObject, NotSet, Opt, is_defined
from github.GithubRetry import GithubRetry
from github.HookDelivery import HookDelivery, HookDeliverySummary
from github.HookDescription import HookDescription
from github.PaginatedList import PaginatedList
from github.RateLimit import RateLimit
from github.Requester import Requester
def get_global_advisories(self, type: Opt[str]=NotSet, ghsa_id: Opt[str]=NotSet, cve_id: Opt[str]=NotSet, ecosystem: Opt[str]=NotSet, severity: Opt[str]=NotSet, cwes: list[Opt[str]] | Opt[str]=NotSet, is_withdrawn: Opt[bool]=NotSet, affects: list[str] | Opt[str]=NotSet, published: Opt[str]=NotSet, updated: Opt[str]=NotSet, modified: Opt[str]=NotSet, keywords: Opt[str]=NotSet, before: Opt[str]=NotSet, after: Opt[str]=NotSet, per_page: Opt[str]=NotSet, sort: Opt[str]=NotSet, direction: Opt[str]=NotSet) -> PaginatedList[GlobalAdvisory]:
    """
        :calls: `GET /advisories <https://docs.github.com/en/rest/security-advisories/global-advisories>`
        :param type: Optional string
        :param ghsa_id: Optional string
        :param cve_id: Optional string
        :param ecosystem: Optional string
        :param severity: Optional string
        :param cwes: Optional comma separated string or list of integer or string
        :param is_withdrawn: Optional bool
        :param affects: Optional comma separated string or list of string
        :param published: Optional string
        :param updated: Optional string
        :param modified: Optional string
        :param before: Optional string
        :param after: Optional string
        :param sort: Optional string
        :param direction: Optional string
        :rtype: :class:`github.PaginatedList.PaginatedList` of :class:`github.GlobalAdvisory.GlobalAdvisory`
        """
    assert type is github.GithubObject.NotSet or isinstance(type, str), type
    assert ghsa_id is github.GithubObject.NotSet or isinstance(ghsa_id, str)
    assert cve_id is github.GithubObject.NotSet or isinstance(cve_id, str), cve_id
    assert ecosystem is github.GithubObject.NotSet or isinstance(ecosystem, str), ecosystem
    assert severity is github.GithubObject.NotSet or isinstance(severity, str), severity
    assert cwes is github.GithubObject.NotSet or isinstance(cwes, list) or isinstance(cwes, str), cwes
    assert is_withdrawn is github.GithubObject.NotSet or isinstance(is_withdrawn, bool), is_withdrawn
    assert affects is github.GithubObject.NotSet or isinstance(affects, list) or isinstance(affects, str), affects
    assert published is github.GithubObject.NotSet or isinstance(published, str), published
    assert updated is github.GithubObject.NotSet or isinstance(updated, str), updated
    assert modified is github.GithubObject.NotSet or isinstance(modified, str), modified
    assert before is github.GithubObject.NotSet or isinstance(before, str), before
    assert after is github.GithubObject.NotSet or isinstance(after, str), after
    assert sort is github.GithubObject.NotSet or isinstance(sort, str), sort
    assert direction is github.GithubObject.NotSet or isinstance(direction, str), direction
    url_parameters: dict[str, Opt[str | bool]] = dict()
    if type is not github.GithubObject.NotSet:
        assert type in ('reviewed', 'unreviewed', 'malware'), type
        url_parameters['type'] = type
    if ghsa_id is not github.GithubObject.NotSet:
        url_parameters['ghsa_id'] = ghsa_id
    if cve_id is not github.GithubObject.NotSet:
        url_parameters['cve_id'] = cve_id
    if ecosystem is not github.GithubObject.NotSet:
        url_parameters['ecosystem'] = ecosystem
    if severity is not github.GithubObject.NotSet:
        assert severity in ('null', 'low', 'medium', 'high', 'critical'), severity
        url_parameters['severity'] = severity
    if cwes is not github.GithubObject.NotSet:
        if isinstance(cwes, list):
            cwes = ','.join([str(cwe) for cwe in cwes])
        url_parameters['cwes'] = cwes
    if is_withdrawn is not github.GithubObject.NotSet:
        url_parameters['is_withdrawn'] = is_withdrawn
    if affects is not github.GithubObject.NotSet:
        if isinstance(affects, list):
            affects = ','.join(affects)
        url_parameters['affects'] = affects
    if published is not github.GithubObject.NotSet:
        url_parameters['published'] = published
    if updated is not github.GithubObject.NotSet:
        url_parameters['updated'] = updated
    if modified is not github.GithubObject.NotSet:
        url_parameters['modified'] = modified
    if before is not github.GithubObject.NotSet:
        url_parameters['before'] = before
    if after is not github.GithubObject.NotSet:
        url_parameters['after'] = after
    if sort is not github.GithubObject.NotSet:
        assert sort in ('published', 'updated'), sort
        url_parameters['sort'] = sort
    if direction is not github.GithubObject.NotSet:
        assert direction in ('asc', 'desc'), direction
        url_parameters['direction'] = direction
    return github.PaginatedList.PaginatedList(github.GlobalAdvisory.GlobalAdvisory, self.__requester, '/advisories', url_parameters)