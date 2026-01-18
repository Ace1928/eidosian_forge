import sys
import joblib
from joblib.testing import check_subprocess_call
from joblib.test.common import with_multiprocessing
@with_multiprocessing
def test_no_semaphore_tracker_on_import():
    code = 'if True:\n        import joblib\n        from multiprocessing import semaphore_tracker\n        # The following line would raise RuntimeError if the\n        # start_method is already set.\n        msg = "multiprocessing.semaphore_tracker has been spawned on import"\n        assert semaphore_tracker._semaphore_tracker._fd is None, msg'
    if sys.version_info >= (3, 8):
        code = code.replace('semaphore_tracker', 'resource_tracker')
    check_subprocess_call([sys.executable, '-c', code])