import sys
import select
def enable_qt5(self, app=None):
    from pydev_ipython.inputhookqt5 import create_inputhook_qt5
    app, inputhook_qt5 = create_inputhook_qt5(self, app)
    self.set_inputhook(inputhook_qt5)
    self._current_gui = GUI_QT5
    app._in_event_loop = True
    self._apps[GUI_QT5] = app
    return app