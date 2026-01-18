import os
import platform
import sys
from functools import partial
import zmq
from packaging.version import Version as V
from traitlets.config.application import Application
@register_integration('wx')
def loop_wx(kernel):
    """Start a kernel with wx event loop support."""
    import wx
    poll_interval = int(1000 * kernel._poll_interval)

    def wake():
        """wake from wx"""
        if kernel.shell_stream.flush(limit=1):
            kernel.app.ExitMainLoop()
            return

    class TimerFrame(wx.Frame):

        def __init__(self, func):
            wx.Frame.__init__(self, None, -1)
            self.timer = wx.Timer(self)
            self.timer.Start(poll_interval)
            self.Bind(wx.EVT_TIMER, self.on_timer)
            self.func = func

        def on_timer(self, event):
            self.func()

    class IPWxApp(wx.App):

        def OnInit(self):
            self.frame = TimerFrame(wake)
            self.frame.Show(False)
            return True
    if not (getattr(kernel, 'app', None) and isinstance(kernel.app, wx.App)):
        kernel.app = IPWxApp(redirect=False)
    import signal
    if not callable(signal.getsignal(signal.SIGINT)):
        signal.signal(signal.SIGINT, signal.default_int_handler)
    _loop_wx(kernel.app)