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
def deployment_branch_policy(self) -> EnvironmentDeploymentBranchPolicy:
    self._completeIfNotSet(self._deployment_branch_policy)
    return self._deployment_branch_policy.value