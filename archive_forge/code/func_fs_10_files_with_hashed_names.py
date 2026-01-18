import os
from hashlib import md5
import pytest
from fsspec.implementations.local import LocalFileSystem
from fsspec.tests.abstract.copy import AbstractCopyTests  # noqa
from fsspec.tests.abstract.get import AbstractGetTests  # noqa
from fsspec.tests.abstract.put import AbstractPutTests  # noqa
@pytest.fixture
def fs_10_files_with_hashed_names(self, fs, fs_join, fs_path):
    """
        Scenario on remote filesystem that is used to check cp/get/put files order
        when source and destination are lists.

        Cleans up at the end of each test it which it is used.
        """
    source = self._10_files_with_hashed_names(fs, fs_join, fs_path)
    yield source
    fs.rm(source, recursive=True)