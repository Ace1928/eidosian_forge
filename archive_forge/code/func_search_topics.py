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
def search_topics(self, query: str, **qualifiers: Any) -> PaginatedList[Topic]:
    """
        :calls: `GET /search/topics <https://docs.github.com/en/rest/reference/search>`_
        :param query: string
        :param qualifiers: keyword dict query qualifiers
        :rtype: :class:`PaginatedList` of :class:`github.Topic.Topic`
        """
    assert isinstance(query, str), query
    url_parameters = dict()
    query_chunks = []
    if query:
        query_chunks.append(query)
    for qualifier, value in qualifiers.items():
        query_chunks.append(f'{qualifier}:{value}')
    url_parameters['q'] = ' '.join(query_chunks)
    assert url_parameters['q'], 'need at least one qualifier'
    return PaginatedList(github.Topic.Topic, self.__requester, '/search/topics', url_parameters, headers={'Accept': Consts.mediaTypeTopicsPreview})