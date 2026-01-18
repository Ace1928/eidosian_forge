from __future__ import annotations
from datetime import datetime
from typing import Any
from github.CVSS import CVSS
from github.CWE import CWE
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def cve_id(self) -> str:
    return self._cve_id.value