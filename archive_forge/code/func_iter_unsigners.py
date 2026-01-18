from __future__ import annotations
import collections.abc as cabc
import json
import typing as t
from .encoding import want_bytes
from .exc import BadPayload
from .exc import BadSignature
from .signer import _make_keys_list
from .signer import Signer
def iter_unsigners(self, salt: str | bytes | None=None) -> cabc.Iterator[Signer]:
    """Iterates over all signers to be tried for unsigning. Starts
        with the configured signer, then constructs each signer
        specified in ``fallback_signers``.
        """
    if salt is None:
        salt = self.salt
    yield self.make_signer(salt)
    for fallback in self.fallback_signers:
        if isinstance(fallback, dict):
            kwargs = fallback
            fallback = self.signer
        elif isinstance(fallback, tuple):
            fallback, kwargs = fallback
        else:
            kwargs = self.signer_kwargs
        for secret_key in self.secret_keys:
            yield fallback(secret_key, salt=salt, **kwargs)