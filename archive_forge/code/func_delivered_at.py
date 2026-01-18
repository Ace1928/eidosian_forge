from datetime import datetime
from typing import Any, Dict, Optional
import github.GithubObject
from github.GithubObject import Attribute, NotSet
@property
def delivered_at(self) -> Optional[datetime]:
    return self._delivered_at.value