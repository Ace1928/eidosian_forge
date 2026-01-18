from __future__ import annotations
import re
import contextlib
import io
import logging
import os
import signal
from subprocess import Popen, PIPE, DEVNULL
import subprocess
import threading
from textwrap import dedent
from git.compat import defenc, force_bytes, safe_decode
from git.exc import (
from git.util import (
from typing import (
from git.types import PathLike, Literal, TBD
@contextlib.contextmanager
def custom_environment(self, **kwargs: Any) -> Iterator[None]:
    """A context manager around the above :meth:`update_environment` method to
        restore the environment back to its previous state after operation.

        ``Examples``::

            with self.custom_environment(GIT_SSH='/bin/ssh_wrapper'):
                repo.remotes.origin.fetch()

        :param kwargs: See :meth:`update_environment`
        """
    old_env = self.update_environment(**kwargs)
    try:
        yield
    finally:
        self.update_environment(**old_env)