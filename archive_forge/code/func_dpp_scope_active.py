import contextlib
from typing import Generator
def dpp_scope_active() -> bool:
    """Returns True if a `dpp_scope` is active. """
    return _dpp_scope_active