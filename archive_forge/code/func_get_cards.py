from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.Project
import github.ProjectCard
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt
from github.PaginatedList import PaginatedList
from . import Consts
def get_cards(self, archived_state: Opt[str]=NotSet) -> PaginatedList[github.ProjectCard.ProjectCard]:
    """
        :calls: `GET /projects/columns/{column_id}/cards <https://docs.github.com/en/rest/reference/projects#list-project-cards>`_
        """
    assert archived_state is NotSet or isinstance(archived_state, str), archived_state
    url_parameters = dict()
    if archived_state is not NotSet:
        url_parameters['archived_state'] = archived_state
    return PaginatedList(github.ProjectCard.ProjectCard, self._requester, f'{self.url}/cards', url_parameters, {'Accept': Consts.mediaTypeProjectsPreview})