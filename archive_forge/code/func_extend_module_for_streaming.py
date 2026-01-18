import importlib
import inspect
from functools import wraps
from typing import TYPE_CHECKING, Optional
from .download.download_config import DownloadConfig
from .download.streaming_download_manager import (
from .utils.logging import get_logger
from .utils.patching import patch_submodule
from .utils.py_utils import get_imports
def extend_module_for_streaming(module_path, download_config: Optional[DownloadConfig]=None):
    """Extend the module to support streaming.

    We patch some functions in the module to use `fsspec` to support data streaming:
    - We use `fsspec.open` to open and read remote files. We patch the module function:
      - `open`
    - We use the "::" hop separator to join paths and navigate remote compressed/archive files. We patch the module
      functions:
      - `os.path.join`
      - `pathlib.Path.joinpath` and `pathlib.Path.__truediv__` (called when using the "/" operator)

    The patched functions are replaced with custom functions defined to work with the
    :class:`~download.streaming_download_manager.StreamingDownloadManager`.

    Args:
        module_path: Path to the module to be extended.
        download_config : mainly use use_auth_token or storage_options to support different platforms and auth types.
    """
    module = importlib.import_module(module_path)
    if hasattr(module, '_patched_for_streaming') and module._patched_for_streaming:
        if isinstance(module._patched_for_streaming, DownloadConfig):
            module._patched_for_streaming.token = download_config.token
            module._patched_for_streaming.storage_options = download_config.storage_options
        return

    def wrap_auth(function):

        @wraps(function)
        def wrapper(*args, **kwargs):
            return function(*args, download_config=download_config, **kwargs)
        wrapper._decorator_name_ = 'wrap_auth'
        return wrapper
    patch_submodule(module, 'open', wrap_auth(xopen)).start()
    patch_submodule(module, 'os.listdir', wrap_auth(xlistdir)).start()
    patch_submodule(module, 'os.walk', wrap_auth(xwalk)).start()
    patch_submodule(module, 'glob.glob', wrap_auth(xglob)).start()
    patch_submodule(module, 'os.path.join', xjoin).start()
    patch_submodule(module, 'os.path.dirname', xdirname).start()
    patch_submodule(module, 'os.path.basename', xbasename).start()
    patch_submodule(module, 'os.path.relpath', xrelpath).start()
    patch_submodule(module, 'os.path.split', xsplit).start()
    patch_submodule(module, 'os.path.splitext', xsplitext).start()
    patch_submodule(module, 'os.path.exists', wrap_auth(xexists)).start()
    patch_submodule(module, 'os.path.isdir', wrap_auth(xisdir)).start()
    patch_submodule(module, 'os.path.isfile', wrap_auth(xisfile)).start()
    patch_submodule(module, 'os.path.getsize', wrap_auth(xgetsize)).start()
    patch_submodule(module, 'pathlib.Path', xPath).start()
    patch_submodule(module, 'gzip.open', wrap_auth(xgzip_open)).start()
    patch_submodule(module, 'numpy.load', wrap_auth(xnumpy_load)).start()
    patch_submodule(module, 'pandas.read_csv', wrap_auth(xpandas_read_csv), attrs=['__version__']).start()
    patch_submodule(module, 'pandas.read_excel', wrap_auth(xpandas_read_excel), attrs=['__version__']).start()
    patch_submodule(module, 'scipy.io.loadmat', wrap_auth(xsio_loadmat), attrs=['__version__']).start()
    patch_submodule(module, 'xml.etree.ElementTree.parse', wrap_auth(xet_parse)).start()
    patch_submodule(module, 'xml.dom.minidom.parse', wrap_auth(xxml_dom_minidom_parse)).start()
    if not module.__name__.startswith('datasets.packaged_modules.'):
        patch_submodule(module, 'pyarrow.parquet.read_table', wrap_auth(xpyarrow_parquet_read_table)).start()
    module._patched_for_streaming = download_config