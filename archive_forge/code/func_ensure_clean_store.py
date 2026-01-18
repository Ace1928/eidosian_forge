from collections.abc import Generator
from contextlib import contextmanager
import pathlib
import tempfile
import pytest
from pandas.io.pytables import HDFStore
@contextmanager
def ensure_clean_store(path, mode='a', complevel=None, complib=None, fletcher32=False) -> Generator[HDFStore, None, None]:
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = pathlib.Path(tmpdirname, path)
        with HDFStore(tmp_path, mode=mode, complevel=complevel, complib=complib, fletcher32=fletcher32) as store:
            yield store