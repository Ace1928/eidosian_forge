from zipfile import ZipFile
import fsspec.utils
from fsspec.spec import AbstractBufferedFile
def isal(infile, mode='rb', **kwargs):
    return igzip.IGzipFile(fileobj=infile, mode=mode, **kwargs)