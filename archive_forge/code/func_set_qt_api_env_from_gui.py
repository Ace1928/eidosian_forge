import os
import platform
import sys
from functools import partial
import zmq
from packaging.version import Version as V
from traitlets.config.application import Application
def set_qt_api_env_from_gui(gui):
    """
    Sets the QT_API environment variable by trying to import PyQtx or PySidex.

    The user can generically request `qt` or a specific Qt version, e.g. `qt6`.
    For a generic Qt request, we let the mechanism in IPython choose the best
    available version by leaving the `QT_API` environment variable blank.

    For specific versions, we check to see whether the PyQt or PySide
    implementations are present and set `QT_API` accordingly to indicate to
    IPython which version we want. If neither implementation is present, we
    leave the environment variable set so IPython will generate a helpful error
    message.

    Notes
    -----
    - If the environment variable is already set, it will be used unchanged,
      regardless of what the user requested.
    """
    qt_api = os.environ.get('QT_API', None)
    from IPython.external.qt_loaders import QT_API_PYQT5, QT_API_PYQT6, QT_API_PYSIDE2, QT_API_PYSIDE6, loaded_api
    loaded = loaded_api()
    qt_env2gui = {QT_API_PYSIDE2: 'qt5', QT_API_PYQT5: 'qt5', QT_API_PYSIDE6: 'qt6', QT_API_PYQT6: 'qt6'}
    if loaded is not None and gui != 'qt' and (qt_env2gui[loaded] != gui):
        print(f'Cannot switch Qt versions for this session; you must use {qt_env2gui[loaded]}.')
        return
    if qt_api is not None and gui != 'qt':
        if qt_env2gui[qt_api] != gui:
            print(f'Request for "{gui}" will be ignored because `QT_API` environment variable is set to "{qt_api}"')
            return
    elif gui == 'qt5':
        try:
            import PyQt5
            os.environ['QT_API'] = 'pyqt5'
        except ImportError:
            try:
                import PySide2
                os.environ['QT_API'] = 'pyside2'
            except ImportError:
                os.environ['QT_API'] = 'pyqt5'
    elif gui == 'qt6':
        try:
            import PyQt6
            os.environ['QT_API'] = 'pyqt6'
        except ImportError:
            try:
                import PySide6
                os.environ['QT_API'] = 'pyside6'
            except ImportError:
                os.environ['QT_API'] = 'pyqt6'
    elif gui == 'qt':
        if 'QT_API' in os.environ:
            del os.environ['QT_API']
    else:
        print(f'Unrecognized Qt version: {gui}. Should be "qt5", "qt6", or "qt".')
        return
    try:
        pass
    except Exception as e:
        if 'QT_API' in os.environ:
            del os.environ['QT_API']
            print(f"QT_API couldn't be set due to error {e}")
        return