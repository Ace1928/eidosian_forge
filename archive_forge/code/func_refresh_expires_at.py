from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Any
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def refresh_expires_at(self) -> datetime | None:
    """
        :type: Optional[datetime]
        """
    seconds = self.refresh_expires_in
    if seconds is not None:
        return self._created + timedelta(seconds=seconds)
    return None