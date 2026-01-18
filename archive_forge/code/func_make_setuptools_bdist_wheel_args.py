import sys
import textwrap
from typing import List, Optional, Sequence
def make_setuptools_bdist_wheel_args(setup_py_path: str, global_options: Sequence[str], build_options: Sequence[str], destination_dir: str) -> List[str]:
    args = make_setuptools_shim_args(setup_py_path, global_options=global_options, unbuffered_output=True)
    args += ['bdist_wheel', '-d', destination_dir]
    args += build_options
    return args