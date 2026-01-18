from __future__ import annotations
import abc
import base64
import json
import os
import tempfile
import typing as t
from ..encoding import (
from ..io import (
from ..config import (
from ..util import (
@cache
def get_ci_provider() -> CIProvider:
    """Return a CI provider instance for the current environment."""
    provider = None
    import_plugins('ci')
    candidates = sorted(get_subclasses(CIProvider), key=lambda subclass: (subclass.priority, subclass.__name__))
    for candidate in candidates:
        if candidate.is_supported():
            provider = candidate()
            break
    if provider.code:
        display.info('Detected CI provider: %s' % provider.name)
    return provider