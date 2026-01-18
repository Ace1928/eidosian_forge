from __future__ import annotations
from .common import CMakeException, CMakeBuildFile, CMakeConfiguration
import typing as T
from .. import mlog
from pathlib import Path
import json
import re
def setup_request(self) -> None:
    self.request_dir.mkdir(parents=True, exist_ok=True)
    query = {'requests': [{'kind': 'codemodel', 'version': {'major': 2, 'minor': 0}}, {'kind': 'cache', 'version': {'major': 2, 'minor': 0}}, {'kind': 'cmakeFiles', 'version': {'major': 1, 'minor': 0}}]}
    query_file = self.request_dir / 'query.json'
    query_file.write_text(json.dumps(query, indent=2), encoding='utf-8')