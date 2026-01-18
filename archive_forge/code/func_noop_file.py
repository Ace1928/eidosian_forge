from zipfile import ZipFile
import fsspec.utils
from fsspec.spec import AbstractBufferedFile
def noop_file(file, mode, **kwargs):
    return file