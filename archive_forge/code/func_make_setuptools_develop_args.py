import sys
import textwrap
from typing import List, Optional, Sequence
def make_setuptools_develop_args(setup_py_path: str, *, global_options: Sequence[str], no_user_config: bool, prefix: Optional[str], home: Optional[str], use_user_site: bool) -> List[str]:
    assert not (use_user_site and prefix)
    args = make_setuptools_shim_args(setup_py_path, global_options=global_options, no_user_config=no_user_config)
    args += ['develop', '--no-deps']
    if prefix:
        args += ['--prefix', prefix]
    if home is not None:
        args += ['--install-dir', home]
    if use_user_site:
        args += ['--user', '--prefix=']
    return args