from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.AdvisoryVulnerabilityPackage
import github.DependabotAlertAdvisory
import github.DependabotAlertDependency
import github.DependabotAlertVulnerability
import github.NamedUser
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def fixed_at(self) -> str | None:
    return self._fixed_at.value