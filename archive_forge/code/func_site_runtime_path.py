from __future__ import annotations
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
@property
def site_runtime_path(self) -> Path:
    """:return: runtime path shared by users"""
    return Path(self.site_runtime_dir)