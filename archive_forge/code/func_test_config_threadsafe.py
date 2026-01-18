import builtins
import time
from concurrent.futures import ThreadPoolExecutor
import pytest
import sklearn
from sklearn import config_context, get_config, set_config
from sklearn.utils import _IS_WASM
from sklearn.utils.parallel import Parallel, delayed
@pytest.mark.xfail(_IS_WASM, reason='cannot start threads')
def test_config_threadsafe():
    """Uses threads directly to test that the global config does not change
    between threads. Same test as `test_config_threadsafe_joblib` but with
    `ThreadPoolExecutor`."""
    assume_finites = [False, True, False, True]
    sleep_durations = [0.1, 0.2, 0.1, 0.2]
    with ThreadPoolExecutor(max_workers=2) as e:
        items = [output for output in e.map(set_assume_finite, assume_finites, sleep_durations)]
    assert items == [False, True, False, True]