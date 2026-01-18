from __future__ import annotations
import collections.abc as cabc
import json
import typing as t
from .encoding import want_bytes
from .exc import BadPayload
from .exc import BadSignature
from .signer import _make_keys_list
from .signer import Signer
def loads_unsafe(self, s: str | bytes, salt: str | bytes | None=None) -> tuple[bool, t.Any]:
    """Like :meth:`loads` but without verifying the signature. This
        is potentially very dangerous to use depending on how your
        serializer works. The return value is ``(signature_valid,
        payload)`` instead of just the payload. The first item will be a
        boolean that indicates if the signature is valid. This function
        never fails.

        Use it for debugging only and if you know that your serializer
        module is not exploitable (for example, do not use it with a
        pickle serializer).

        .. versionadded:: 0.15
        """
    return self._loads_unsafe_impl(s, salt)