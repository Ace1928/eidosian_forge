from __future__ import annotations
import abc
import os
import typing as t
from ..util import (
def get_path_provider_classes(provider_type: t.Type[TPathProvider]) -> list[t.Type[TPathProvider]]:
    """Return a list of path provider classes of the given type."""
    return sorted(get_subclasses(provider_type), key=lambda subclass: (subclass.priority, subclass.__name__))