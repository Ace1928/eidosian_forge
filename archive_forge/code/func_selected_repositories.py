from datetime import datetime
from typing import Any, Dict
from github.GithubObject import Attribute, NotSet
from github.PaginatedList import PaginatedList
from github.Repository import Repository
from github.Secret import Secret
@property
def selected_repositories(self) -> PaginatedList[Repository]:
    return PaginatedList(Repository, self._requester, self._selected_repositories_url.value, None, list_item='repositories')