from __future__ import annotations
from .common import CMakeException, CMakeBuildFile, CMakeConfiguration
import typing as T
from .. import mlog
from pathlib import Path
import json
import re
def get_cmake_sources(self) -> T.List[CMakeBuildFile]:
    return self.cmake_sources