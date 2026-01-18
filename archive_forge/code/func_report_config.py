from __future__ import annotations
from .base import ExternalDependency, DependencyException, DependencyTypeName
from ..mesonlib import listify, Popen_safe, Popen_safe_logged, split_args, version_compare, version_compare_many
from ..programs import find_external_program
from .. import mlog
import re
import typing as T
from mesonbuild import mesonlib
def report_config(self, version: T.Optional[str], req_version: T.List[str]) -> bool:
    """Helper method to print messages about the tool."""
    found_msg: T.List[T.Union[str, mlog.AnsiDecorator]] = [mlog.bold(self.tool_name), 'found:']
    if self.config is None:
        found_msg.append(mlog.red('NO'))
        if version is not None and req_version:
            found_msg.append(f'found {version!r} but need {req_version!r}')
        elif req_version:
            found_msg.append(f'need {req_version!r}')
    else:
        found_msg += [mlog.green('YES'), '({})'.format(' '.join(self.config)), version]
    mlog.log(*found_msg)
    return self.config is not None