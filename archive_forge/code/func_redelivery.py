from datetime import datetime
from typing import Any, Dict, Optional
import github.GithubObject
from github.GithubObject import Attribute, NotSet
@property
def redelivery(self) -> Optional[bool]:
    return self._redelivery.value