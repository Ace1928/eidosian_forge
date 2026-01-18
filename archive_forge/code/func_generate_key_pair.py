from __future__ import annotations
import abc
import dataclasses
import json
import os
import re
import stat
import traceback
import uuid
import time
import typing as t
from .http import (
from .io import (
from .util import (
from .util_common import (
from .config import (
from .ci import (
from .data import (
def generate_key_pair(self, args: EnvironmentConfig) -> tuple[str, str]:
    """Generate an SSH key pair for use by all ansible-test invocations for the current user."""
    key, pub = self.get_source_key_pair_paths()
    if not args.explain:
        make_dirs(os.path.dirname(key))
    if not os.path.isfile(key) or not os.path.isfile(pub):
        run_command(args, ['ssh-keygen', '-m', 'PEM', '-q', '-t', self.KEY_TYPE, '-N', '', '-f', key], capture=True)
        if args.explain:
            return (key, pub)
        key_contents = read_text_file(key)
        key_contents = re.sub('(BEGIN|END) PRIVATE KEY', '\\1 RSA PRIVATE KEY', key_contents)
        write_text_file(key, key_contents)
    return (key, pub)