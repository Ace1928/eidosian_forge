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
def get_project_column(self, id: int) -> ProjectColumn:
    """
        :calls: `GET /projects/columns/{column_id} <https://docs.github.com/en/rest/reference/projects#get-a-project-column>`_
        """
    headers, data = self.__requester.requestJsonAndCheck('GET', '/projects/columns/%d' % id, headers={'Accept': Consts.mediaTypeProjectsPreview})
    return github.ProjectColumn.ProjectColumn(self.__requester, headers, data, completed=True)