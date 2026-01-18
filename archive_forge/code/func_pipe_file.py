import zipfile
import fsspec
from fsspec.archive import AbstractArchiveFileSystem
def pipe_file(self, path, value, **kwargs):
    self.zip.writestr(path, value, **kwargs)