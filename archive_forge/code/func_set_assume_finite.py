import builtins
import time
from concurrent.futures import ThreadPoolExecutor
import pytest
import sklearn
from sklearn import config_context, get_config, set_config
from sklearn.utils import _IS_WASM
from sklearn.utils.parallel import Parallel, delayed
def set_assume_finite(assume_finite, sleep_duration):
    """Return the value of assume_finite after waiting `sleep_duration`."""
    with config_context(assume_finite=assume_finite):
        time.sleep(sleep_duration)
        return get_config()['assume_finite']