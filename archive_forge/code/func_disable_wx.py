import sys
import select
def disable_wx(self):
    """Disable event loop integration with wxPython.

        This merely sets PyOS_InputHook to NULL.
        """
    if GUI_WX in self._apps:
        self._apps[GUI_WX]._in_event_loop = False
    self.clear_inputhook()