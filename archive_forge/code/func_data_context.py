from __future__ import annotations
import collections.abc as c
import dataclasses
import os
import typing as t
from .util import (
from .provider import (
from .provider.source import (
from .provider.source.unversioned import (
from .provider.source.installed import (
from .provider.source.unsupported import (
from .provider.layout import (
from .provider.layout.unsupported import (
@cache
def data_context() -> DataContext:
    """Initialize provider plugins."""
    provider_types = ('layout', 'source')
    for provider_type in provider_types:
        import_plugins('provider/%s' % provider_type)
    context = DataContext()
    return context