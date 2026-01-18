import sys
import textwrap
from typing import List, Optional, Sequence
def make_setuptools_shim_args(setup_py_path: str, global_options: Optional[Sequence[str]]=None, no_user_config: bool=False, unbuffered_output: bool=False) -> List[str]:
    """
    Get setuptools command arguments with shim wrapped setup file invocation.

    :param setup_py_path: The path to setup.py to be wrapped.
    :param global_options: Additional global options.
    :param no_user_config: If True, disables personal user configuration.
    :param unbuffered_output: If True, adds the unbuffered switch to the
     argument list.
    """
    args = [sys.executable]
    if unbuffered_output:
        args += ['-u']
    args += ['-c', _SETUPTOOLS_SHIM.format(setup_py_path)]
    if global_options:
        args += global_options
    if no_user_config:
        args += ['--no-user-cfg']
    return args