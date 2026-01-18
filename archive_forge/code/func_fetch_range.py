import dask
from distributed.client import Client, _get_global_client
from distributed.worker import Worker
from fsspec import filesystem
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
from fsspec.utils import infer_storage_options
def fetch_range(self, path, mode, start, end):
    if self.worker:
        with self._open(path, mode) as f:
            f.seek(start)
            return f.read(end - start)
    else:
        return self.rfs.fetch_range(path, mode, start, end).compute()