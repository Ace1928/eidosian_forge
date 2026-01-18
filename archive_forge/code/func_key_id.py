from __future__ import annotations
from base64 import b64encode
from typing import Any
from nacl import encoding, public
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def key_id(self) -> str | int:
    self._completeIfNotSet(self._key_id)
    return self._key_id.value