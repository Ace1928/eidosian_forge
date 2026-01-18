import sys
import select
def enable_wx(self, app=None):
    """Enable event loop integration with wxPython.

        Parameters
        ----------
        app : WX Application, optional.
            Running application to use.  If not given, we probe WX for an
            existing application object, and create a new one if none is found.

        Notes
        -----
        This methods sets the ``PyOS_InputHook`` for wxPython, which allows
        the wxPython to integrate with terminal based applications like
        IPython.

        If ``app`` is not given we probe for an existing one, and return it if
        found.  If no existing app is found, we create an :class:`wx.App` as
        follows::

            import wx
            app = wx.App(redirect=False, clearSigInt=False)
        """
    import wx
    from distutils.version import LooseVersion as V
    wx_version = V(wx.__version__).version
    if wx_version < [2, 8]:
        raise ValueError('requires wxPython >= 2.8, but you have %s' % wx.__version__)
    from pydev_ipython.inputhookwx import inputhook_wx
    self.set_inputhook(inputhook_wx)
    self._current_gui = GUI_WX
    if app is None:
        app = wx.GetApp()
    if app is None:
        app = wx.App(redirect=False, clearSigInt=False)
    app._in_event_loop = True
    self._apps[GUI_WX] = app
    return app