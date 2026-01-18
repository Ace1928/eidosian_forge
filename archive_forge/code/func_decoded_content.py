from __future__ import annotations
import base64
from typing import TYPE_CHECKING, Any
import github.GithubObject
import github.Repository
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, _ValuedAttribute
@property
def decoded_content(self) -> bytes:
    assert self.encoding == 'base64', f'unsupported encoding: {self.encoding}'
    return base64.b64decode(bytearray(self.content, 'utf-8'))