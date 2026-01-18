from __future__ import annotations
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
@property
def site_config_path(self) -> Path:
    """:return: config path shared by the users"""
    return Path(self.site_config_dir)