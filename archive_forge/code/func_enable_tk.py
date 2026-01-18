import sys
import select
def enable_tk(self, app=None):
    """Enable event loop integration with Tk.

        Parameters
        ----------
        app : toplevel :class:`Tkinter.Tk` widget, optional.
            Running toplevel widget to use.  If not given, we probe Tk for an
            existing one, and create a new one if none is found.

        Notes
        -----
        If you have already created a :class:`Tkinter.Tk` object, the only
        thing done by this method is to register with the
        :class:`InputHookManager`, since creating that object automatically
        sets ``PyOS_InputHook``.
        """
    self._current_gui = GUI_TK
    if app is None:
        try:
            import Tkinter as _TK
        except:
            import tkinter as _TK
        app = _TK.Tk()
        app.withdraw()
        self._apps[GUI_TK] = app
    from pydev_ipython.inputhooktk import create_inputhook_tk
    self.set_inputhook(create_inputhook_tk(app))
    return app