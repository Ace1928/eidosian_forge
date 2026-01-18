import builtins
import time
from concurrent.futures import ThreadPoolExecutor
import pytest
import sklearn
from sklearn import config_context, get_config, set_config
from sklearn.utils import _IS_WASM
from sklearn.utils.parallel import Parallel, delayed
def mocked_import(name, *args, **kwargs):
    if name == 'array_api_compat':
        return object()
    return orig_import(name, *args, **kwargs)