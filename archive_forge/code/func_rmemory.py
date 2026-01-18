import contextlib
import typing
from . import openrlib
from _cffi_backend import FFI  # type: ignore
@contextlib.contextmanager
def rmemory() -> typing.Iterator[ProtectionTracker]:
    pt = ProtectionTracker()
    with openrlib.rlock:
        try:
            yield pt
        finally:
            pt.unprotect_all()