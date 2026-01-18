from zipfile import ZipFile
import fsspec.utils
from fsspec.spec import AbstractBufferedFile
def available_compressions():
    """Return a list of the implemented compressions."""
    return list(compr)