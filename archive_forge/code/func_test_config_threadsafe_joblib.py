import builtins
import time
from concurrent.futures import ThreadPoolExecutor
import pytest
import sklearn
from sklearn import config_context, get_config, set_config
from sklearn.utils import _IS_WASM
from sklearn.utils.parallel import Parallel, delayed
@pytest.mark.parametrize('backend', ['loky', 'multiprocessing', 'threading'])
def test_config_threadsafe_joblib(backend):
    """Test that the global config is threadsafe with all joblib backends.
    Two jobs are spawned and sets assume_finite to two different values.
    When the job with a duration 0.1s completes, the assume_finite value
    should be the same as the value passed to the function. In other words,
    it is not influenced by the other job setting assume_finite to True.
    """
    assume_finites = [False, True, False, True]
    sleep_durations = [0.1, 0.2, 0.1, 0.2]
    items = Parallel(backend=backend, n_jobs=2)((delayed(set_assume_finite)(assume_finite, sleep_dur) for assume_finite, sleep_dur in zip(assume_finites, sleep_durations)))
    assert items == [False, True, False, True]