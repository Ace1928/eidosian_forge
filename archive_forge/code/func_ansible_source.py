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
@property
def ansible_source(self) -> tuple[tuple[str, str], ...]:
    """Return a tuple of Ansible source files with both absolute and relative paths."""
    if not self.__ansible_source:
        self.__ansible_source = self.__create_ansible_source()
    return self.__ansible_source