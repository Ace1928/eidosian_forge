from datetime import datetime
from typing import Any, Dict
import github.GithubObject
from github.GithubObject import Attribute
@property
def days(self) -> int:
    return self._days.value