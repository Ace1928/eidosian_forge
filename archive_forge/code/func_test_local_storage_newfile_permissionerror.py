import os
import shutil
import time
from pathlib import Path
import tempfile
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pytest
from ..utils import (
def test_local_storage_newfile_permissionerror(monkeypatch):
    """Should warn the user when can't write to the local data dir"""

    def mocktempfile(**kwargs):
        """Raise an exception to mimic permission issues"""
        raise PermissionError('Fake error')
    with TemporaryDirectory() as data_cache:
        os.makedirs(os.path.join(data_cache, '1.0'))
        assert os.path.exists(data_cache)
        monkeypatch.setattr(tempfile, 'NamedTemporaryFile', mocktempfile)
        with pytest.raises(PermissionError) as error:
            make_local_storage(path=data_cache, env='SOME_VARIABLE')
            assert 'Pooch could not write to data cache' in str(error)
            assert "'SOME_VARIABLE'" in str(error)