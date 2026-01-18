import sys
import select
def enable_mac(self, app=None):
    """ Enable event loop integration with MacOSX.

        We call function pyplot.pause, which updates and displays active
        figure during pause. It's not MacOSX-specific, but it enables to
        avoid inputhooks in native MacOSX backend.
        Also we shouldn't import pyplot, until user does it. Cause it's
        possible to choose backend before importing pyplot for the first
        time only.
        """

    def inputhook_mac(app=None):
        if self.pyplot_imported:
            pyplot = sys.modules['matplotlib.pyplot']
            try:
                pyplot.pause(0.01)
            except:
                pass
        elif 'matplotlib.pyplot' in sys.modules:
            self.pyplot_imported = True
    self.set_inputhook(inputhook_mac)
    self._current_gui = GUI_OSX