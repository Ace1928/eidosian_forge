from __future__ import annotations
from .common import CMakeException, CMakeBuildFile, CMakeConfiguration
import typing as T
from .. import mlog
from pathlib import Path
import json
import re
def load_reply(self) -> None:
    if not self.reply_dir.is_dir():
        raise CMakeException('No response from the CMake file API')
    root = None
    reg_index = re.compile('^index-.*\\.json$')
    for i in self.reply_dir.iterdir():
        if reg_index.match(i.name):
            root = i
            break
    if not root:
        raise CMakeException('Failed to find the CMake file API index')
    index = self._reply_file_content(root)
    index = self._strip_data(index)
    index = self._resolve_references(index)
    index = self._strip_data(index)
    debug_json = self.build_dir / '..' / 'fileAPI.json'
    debug_json = debug_json.resolve()
    debug_json.write_text(json.dumps(index, indent=2), encoding='utf-8')
    mlog.cmd_ci_include(debug_json.as_posix())
    for i in index['objects']:
        assert isinstance(i, dict)
        assert 'kind' in i
        assert i['kind'] in self.kind_resolver_map
        self.kind_resolver_map[i['kind']](i)