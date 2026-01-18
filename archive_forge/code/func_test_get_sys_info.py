from sklearn.utils._show_versions import _get_deps_info, _get_sys_info, show_versions
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.fixes import threadpool_info
def test_get_sys_info():
    sys_info = _get_sys_info()
    assert 'python' in sys_info
    assert 'executable' in sys_info
    assert 'machine' in sys_info