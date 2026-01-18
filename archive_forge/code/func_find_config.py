from __future__ import annotations
from .base import ExternalDependency, DependencyException, DependencyTypeName
from ..mesonlib import listify, Popen_safe, Popen_safe_logged, split_args, version_compare, version_compare_many
from ..programs import find_external_program
from .. import mlog
import re
import typing as T
from mesonbuild import mesonlib
def find_config(self, versions: T.List[str], returncode: int=0) -> T.Tuple[T.Optional[T.List[str]], T.Optional[str]]:
    """Helper method that searches for config tool binaries in PATH and
        returns the one that best matches the given version requirements.
        """
    best_match: T.Tuple[T.Optional[T.List[str]], T.Optional[str]] = (None, None)
    for potential_bin in find_external_program(self.env, self.for_machine, self.tool_name, self.tool_name, self.tools, allow_default_for_cross=self.allow_default_for_cross):
        if not potential_bin.found():
            continue
        tool = potential_bin.get_command()
        try:
            p, out = Popen_safe(tool + [self.version_arg])[:2]
        except (FileNotFoundError, PermissionError):
            continue
        if p.returncode != returncode:
            if self.skip_version:
                p = Popen_safe(tool + [self.skip_version])[0]
                if p.returncode != returncode:
                    continue
            else:
                continue
        out = self._sanitize_version(out.strip())
        if not out:
            return (tool, None)
        if versions:
            is_found = version_compare_many(out, versions)[0]
            if not is_found:
                tool = None
        if best_match[1]:
            if version_compare(out, '> {}'.format(best_match[1])):
                best_match = (tool, out)
        else:
            best_match = (tool, out)
    return best_match