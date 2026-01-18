import sys
import joblib
from joblib.testing import check_subprocess_call
from joblib.test.common import with_multiprocessing
@with_multiprocessing
def test_no_resource_tracker_on_import():
    code = 'if True:\n        import joblib\n        from joblib.externals.loky.backend import resource_tracker\n        # The following line would raise RuntimeError if the\n        # start_method is already set.\n        msg = "loky.resource_tracker has been spawned on import"\n        assert resource_tracker._resource_tracker._fd is None, msg\n    '
    check_subprocess_call([sys.executable, '-c', code])