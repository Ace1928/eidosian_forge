from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.WorkflowRun
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def archive_download_url(self) -> str:
    return self._archive_download_url.value