from __future__ import annotations
from typing import TYPE_CHECKING, Any, Union
from typing_extensions import TypedDict
import github.AdvisoryVulnerabilityPackage
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def patched_versions(self) -> str:
    """
        :type: string
        """
    return self._patched_versions.value