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
def security_advisory(self) -> DependabotAlertAdvisory:
    return self._security_advisory.value