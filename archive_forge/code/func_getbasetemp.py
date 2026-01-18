import dataclasses
import os
from pathlib import Path
import re
from shutil import rmtree
import tempfile
from typing import Any
from typing import Dict
from typing import final
from typing import Generator
from typing import Literal
from typing import Optional
from typing import Union
from .pathlib import cleanup_dead_symlinks
from .pathlib import LOCK_TIMEOUT
from .pathlib import make_numbered_dir
from .pathlib import make_numbered_dir_with_cleanup
from .pathlib import rm_rf
from _pytest.compat import get_user_id
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import fixture
from _pytest.fixtures import FixtureRequest
from _pytest.monkeypatch import MonkeyPatch
from _pytest.nodes import Item
from _pytest.reports import TestReport
from _pytest.stash import StashKey
def getbasetemp(self) -> Path:
    """Return the base temporary directory, creating it if needed.

        :returns:
            The base temporary directory.
        """
    if self._basetemp is not None:
        return self._basetemp
    if self._given_basetemp is not None:
        basetemp = self._given_basetemp
        if basetemp.exists():
            rm_rf(basetemp)
        basetemp.mkdir(mode=448)
        basetemp = basetemp.resolve()
    else:
        from_env = os.environ.get('PYTEST_DEBUG_TEMPROOT')
        temproot = Path(from_env or tempfile.gettempdir()).resolve()
        user = get_user() or 'unknown'
        rootdir = temproot.joinpath(f'pytest-of-{user}')
        try:
            rootdir.mkdir(mode=448, exist_ok=True)
        except OSError:
            rootdir = temproot.joinpath('pytest-of-unknown')
            rootdir.mkdir(mode=448, exist_ok=True)
        uid = get_user_id()
        if uid is not None:
            rootdir_stat = rootdir.stat()
            if rootdir_stat.st_uid != uid:
                raise OSError(f'The temporary directory {rootdir} is not owned by the current user. Fix this and try again.')
            if rootdir_stat.st_mode & 63 != 0:
                os.chmod(rootdir, rootdir_stat.st_mode & ~63)
        keep = self._retention_count
        if self._retention_policy == 'none':
            keep = 0
        basetemp = make_numbered_dir_with_cleanup(prefix='pytest-', root=rootdir, keep=keep, lock_timeout=LOCK_TIMEOUT, mode=448)
    assert basetemp is not None, basetemp
    self._basetemp = basetemp
    self._trace('new basetemp', basetemp)
    return basetemp