from __future__ import annotations
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
def iter_data_dirs(self) -> Iterator[str]:
    """:yield: all user and site data directories."""
    yield self.user_data_dir
    yield self.site_data_dir