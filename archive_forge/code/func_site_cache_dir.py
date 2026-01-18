from __future__ import annotations
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
@property
@abstractmethod
def site_cache_dir(self) -> str:
    """:return: cache directory shared by users"""