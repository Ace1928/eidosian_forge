import sys
import textwrap
from typing import List, Optional, Sequence
def make_setuptools_egg_info_args(setup_py_path: str, egg_info_dir: Optional[str], no_user_config: bool) -> List[str]:
    args = make_setuptools_shim_args(setup_py_path, no_user_config=no_user_config)
    args += ['egg_info']
    if egg_info_dir:
        args += ['--egg-base', egg_info_dir]
    return args