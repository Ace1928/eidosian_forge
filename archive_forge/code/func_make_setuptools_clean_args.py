import sys
import textwrap
from typing import List, Optional, Sequence
def make_setuptools_clean_args(setup_py_path: str, global_options: Sequence[str]) -> List[str]:
    args = make_setuptools_shim_args(setup_py_path, global_options=global_options, unbuffered_output=True)
    args += ['clean', '--all']
    return args