import builtins
import time
from concurrent.futures import ThreadPoolExecutor
import pytest
import sklearn
from sklearn import config_context, get_config, set_config
from sklearn.utils import _IS_WASM
from sklearn.utils.parallel import Parallel, delayed
def test_config_array_api_dispatch_error(monkeypatch):
    """Check error is raised when array_api_compat is not installed."""
    orig_import = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name == 'array_api_compat':
            raise ImportError
        return orig_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', mocked_import)
    with pytest.raises(ImportError, match='array_api_compat is required'):
        with config_context(array_api_dispatch=True):
            pass
    with pytest.raises(ImportError, match='array_api_compat is required'):
        set_config(array_api_dispatch=True)