import pytest
import sys
import matplotlib
from matplotlib import _api
@pytest.fixture(autouse=True)
def mpl_test_settings(request):
    from matplotlib.testing.decorators import _cleanup_cm
    with _cleanup_cm():
        backend = None
        backend_marker = request.node.get_closest_marker('backend')
        prev_backend = matplotlib.get_backend()
        if backend_marker is not None:
            assert len(backend_marker.args) == 1, "Marker 'backend' must specify 1 backend."
            backend, = backend_marker.args
            skip_on_importerror = backend_marker.kwargs.get('skip_on_importerror', False)
            if backend.lower().startswith('qt5'):
                if any((sys.modules.get(k) for k in ('PyQt4', 'PySide'))):
                    pytest.skip('Qt4 binding already imported')
        matplotlib.testing.setup()
        with _api.suppress_matplotlib_deprecation_warning():
            if backend is not None:
                import matplotlib.pyplot as plt
                try:
                    plt.switch_backend(backend)
                except ImportError as exc:
                    if 'cairo' in backend.lower() or skip_on_importerror:
                        pytest.skip(f'Failed to switch to backend {backend} ({exc}).')
                    else:
                        raise
            matplotlib.style.use(['classic', '_classic_test_patch'])
        try:
            yield
        finally:
            if backend is not None:
                plt.close('all')
                matplotlib.use(prev_backend)