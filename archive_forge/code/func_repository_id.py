from datetime import datetime
from typing import Any, Dict, Optional
import github.GithubObject
from github.GithubObject import Attribute, NotSet
@property
def repository_id(self) -> Optional[int]:
    return self._repository_id.value