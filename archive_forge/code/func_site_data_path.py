from __future__ import annotations
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
@property
def site_data_path(self) -> Path:
    """:return: data path shared by users"""
    return Path(self.site_data_dir)