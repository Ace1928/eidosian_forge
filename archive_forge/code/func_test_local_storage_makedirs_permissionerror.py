import os
import shutil
import time
from pathlib import Path
import tempfile
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pytest
from ..utils import (
def test_local_storage_makedirs_permissionerror(monkeypatch):
    """Should warn the user when can't create the local data dir"""

    def mockmakedirs(path, exist_ok=False):
        """Raise an exception to mimic permission issues"""
        raise PermissionError('Fake error')
    data_cache = os.path.join(os.curdir, 'test_permission')
    assert not os.path.exists(data_cache)
    monkeypatch.setattr(os, 'makedirs', mockmakedirs)
    with pytest.raises(PermissionError) as error:
        make_local_storage(path=data_cache, env='SOME_VARIABLE')
        assert 'Pooch could not create data cache' in str(error)
        assert "'SOME_VARIABLE'" in str(error)