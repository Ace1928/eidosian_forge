from __future__ import annotations
import collections.abc as c
import contextlib
import json
import os
import re
import shlex
import sys
import tempfile
import textwrap
import typing as t
from .constants import (
from .encoding import (
from .util import (
from .io import (
from .data import (
from .provider.layout import (
from .host_configs import (
@cache
def get_injector_path() -> str:
    """Return the path to a directory which contains a `python.py` executable and associated injector scripts."""
    injector_path = tempfile.mkdtemp(prefix='ansible-test-', suffix='-injector', dir='/tmp')
    display.info(f'Initializing "{injector_path}" as the temporary injector directory.', verbosity=1)
    injector_names = sorted(list(ANSIBLE_BIN_SYMLINK_MAP) + ['importer.py', 'pytest'])
    scripts = (('python.py', '/usr/bin/env python', MODE_FILE_EXECUTE), ('virtualenv.sh', '/usr/bin/env bash', MODE_FILE))
    source_path = os.path.join(ANSIBLE_TEST_TARGET_ROOT, 'injector')
    for name in injector_names:
        os.symlink('python.py', os.path.join(injector_path, name))
    for name, shebang, mode in scripts:
        src = os.path.join(source_path, name)
        dst = os.path.join(injector_path, name)
        script = read_text_file(src)
        script = set_shebang(script, shebang)
        write_text_file(dst, script)
        verified_chmod(dst, mode)
    verified_chmod(injector_path, MODE_DIRECTORY)

    def cleanup_injector() -> None:
        """Remove the temporary injector directory."""
        remove_tree(injector_path)
    ExitHandler.register(cleanup_injector)
    return injector_path