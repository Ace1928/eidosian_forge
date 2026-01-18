from __future__ import annotations
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
def iter_runtime_dirs(self) -> Iterator[str]:
    """:yield: all user and site runtime directories."""
    yield self.user_runtime_dir
    yield self.site_runtime_dir