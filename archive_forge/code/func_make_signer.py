from __future__ import annotations
import collections.abc as cabc
import json
import typing as t
from .encoding import want_bytes
from .exc import BadPayload
from .exc import BadSignature
from .signer import _make_keys_list
from .signer import Signer
def make_signer(self, salt: str | bytes | None=None) -> Signer:
    """Creates a new instance of the signer to be used. The default
        implementation uses the :class:`.Signer` base class.
        """
    if salt is None:
        salt = self.salt
    return self.signer(self.secret_keys, salt=salt, **self.signer_kwargs)