from __future__ import annotations
import collections.abc as cabc
import json
import typing as t
from .encoding import want_bytes
from .exc import BadPayload
from .exc import BadSignature
from .signer import _make_keys_list
from .signer import Signer
def load_unsafe(self, f: t.IO[t.Any], salt: str | bytes | None=None) -> tuple[bool, t.Any]:
    """Like :meth:`loads_unsafe` but loads from a file.

        .. versionadded:: 0.15
        """
    return self.loads_unsafe(f.read(), salt=salt)