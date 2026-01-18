from __future__ import annotations
from typing import Any
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def ecosystem(self) -> str:
    """
        :type: string
        """
    return self._ecosystem.value