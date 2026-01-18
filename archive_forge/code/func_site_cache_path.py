from __future__ import annotations
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
@property
def site_cache_path(self) -> Path:
    """:return: cache path shared by users"""
    return Path(self.site_cache_dir)