from __future__ import annotations
import io
import typing as t
from functools import partial
from functools import update_wrapper
from .exceptions import ClientDisconnected
from .exceptions import RequestEntityTooLarge
from .sansio import utils as _sansio_utils
from .sansio.utils import host_is_trusted  # noqa: F401 # Imported as part of API
def on_exhausted(self) -> None:
    """Called when attempting to read after the limit has been reached.

        The default behavior is to do nothing, unless the limit is a maximum, in which
        case it raises :exc:`.RequestEntityTooLarge`.

        .. versionchanged:: 2.3
            Raises ``RequestEntityTooLarge`` if the limit is a maximum.

        .. versionchanged:: 2.3
            Any return value is ignored.
        """
    if self._limit_is_max:
        raise RequestEntityTooLarge()