from __future__ import annotations
import json
import os
import pathlib
import pickle
import re
import sys
import typing as T
from ..backend.ninjabackend import ninja_quote
from ..compilers.compilers import lang_suffixes
def scan_cpp_file(self, fname: str) -> None:
    fpath = pathlib.Path(fname)
    for line in fpath.read_text(encoding='utf-8', errors='ignore').split('\n'):
        import_match = CPP_IMPORT_RE.match(line)
        export_match = CPP_EXPORT_RE.match(line)
        if import_match:
            needed = import_match.group(1)
            if fname in self.needs:
                self.needs[fname].append(needed)
            else:
                self.needs[fname] = [needed]
        if export_match:
            exported_module = export_match.group(1)
            if exported_module in self.provided_by:
                raise RuntimeError(f'Multiple files provide module {exported_module}.')
            self.sources_with_exports.append(fname)
            self.provided_by[exported_module] = fname
            self.exports[fname] = exported_module