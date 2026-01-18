import sys
import select
def disable_qt5(self):
    if GUI_QT5 in self._apps:
        self._apps[GUI_QT5]._in_event_loop = False
    self.clear_inputhook()