from __future__ import annotations
import shlex
import subprocess
import sys
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Union
from .. import util
from ..util import compat
@register('console_scripts')
def console_scripts(path: str, options: dict, ignore_output: bool=False) -> None:
    try:
        entrypoint_name = options['entrypoint']
    except KeyError as ke:
        raise util.CommandError(f'Key {options['_hook_name']}.entrypoint is required for post write hook {options['_hook_name']!r}') from ke
    for entry in compat.importlib_metadata_get('console_scripts'):
        if entry.name == entrypoint_name:
            impl: Any = entry
            break
    else:
        raise util.CommandError(f'Could not find entrypoint console_scripts.{entrypoint_name}')
    cwd: Optional[str] = options.get('cwd', None)
    cmdline_options_str = options.get('options', '')
    cmdline_options_list = _parse_cmdline_options(cmdline_options_str, path)
    kw: Dict[str, Any] = {}
    if ignore_output:
        kw['stdout'] = kw['stderr'] = subprocess.DEVNULL
    subprocess.run([sys.executable, '-c', f'import {impl.module}; {impl.module}.{impl.attr}()'] + cmdline_options_list, cwd=cwd, **kw)