import sys
from functools import partial
from pydev_ipython.version import check_version
def load_qt(api_options):
    """
    Attempt to import Qt, given a preference list
    of permissible bindings

    It is safe to call this function multiple times.

    Parameters
    ----------
    api_options: List of strings
        The order of APIs to try. Valid items are 'pyside',
        'pyqt', and 'pyqtv1'

    Returns
    -------

    A tuple of QtCore, QtGui, QtSvg, QT_API
    The first three are the Qt modules. The last is the
    string indicating which module was loaded.

    Raises
    ------
    ImportError, if it isn't possible to import any requested
    bindings (either becaues they aren't installed, or because
    an incompatible library has already been installed)
    """
    loaders = {QT_API_PYSIDE: import_pyside, QT_API_PYSIDE2: import_pyside2, QT_API_PYQT: import_pyqt4, QT_API_PYQTv1: partial(import_pyqt4, version=1), QT_API_PYQT_DEFAULT: partial(import_pyqt4, version=None), QT_API_PYQT5: import_pyqt5}
    for api in api_options:
        if api not in loaders:
            raise RuntimeError('Invalid Qt API %r, valid values are: %r, %r, %r, %r, %r, %r' % (api, QT_API_PYSIDE, QT_API_PYSIDE, QT_API_PYQT, QT_API_PYQTv1, QT_API_PYQT_DEFAULT, QT_API_PYQT5))
        if not can_import(api):
            continue
        result = loaders[api]()
        api = result[-1]
        commit_api(api)
        return result
    else:
        raise ImportError('\n    Could not load requested Qt binding. Please ensure that\n    PyQt4 >= 4.7 or PySide >= 1.0.3 is available,\n    and only one is imported per session.\n\n    Currently-imported Qt library:   %r\n    PyQt4 installed:                 %s\n    PyQt5 installed:                 %s\n    PySide >= 1.0.3 installed:       %s\n    PySide2 installed:               %s\n    Tried to load:                   %r\n    ' % (loaded_api(), has_binding(QT_API_PYQT), has_binding(QT_API_PYQT5), has_binding(QT_API_PYSIDE), has_binding(QT_API_PYSIDE2), api_options))