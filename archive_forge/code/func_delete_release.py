from __future__ import annotations
from datetime import datetime
from os.path import basename
from typing import Any, BinaryIO
import github.GitReleaseAsset
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt
from github.PaginatedList import PaginatedList
from . import Consts
def delete_release(self) -> None:
    """
        :calls: `DELETE /repos/{owner}/{repo}/releases/{release_id} <https://docs.github.com/en/rest/reference/repos#delete-a-release>`_
        """
    headers, data = self._requester.requestJsonAndCheck('DELETE', self.url)