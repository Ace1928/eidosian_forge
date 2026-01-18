from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.EnvironmentDeploymentBranchPolicy
import github.EnvironmentProtectionRule
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
from github.PaginatedList import PaginatedList
from github.PublicKey import PublicKey
from github.Secret import Secret
from github.Variable import Variable
@property
def protection_rules(self) -> list[EnvironmentProtectionRule]:
    self._completeIfNotSet(self._protection_rules)
    return self._protection_rules.value