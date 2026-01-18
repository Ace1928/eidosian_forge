from IPython.core.getipython import get_ipython
def get_app_qt4(*args, **kwargs):
    """Create a new Qt app or return an existing one."""
    from IPython.external.qt_for_kernel import QtGui
    app = QtGui.QApplication.instance()
    if app is None:
        if not args:
            args = ([''],)
        app = QtGui.QApplication(*args, **kwargs)
    return app