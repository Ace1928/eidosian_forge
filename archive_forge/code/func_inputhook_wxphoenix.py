import sys
import signal
import time
from timeit import default_timer as clock
import wx
@ignore_keyboardinterrupts
def inputhook_wxphoenix(context):
    """Run the wx event loop until the user provides more input.

    This input hook is suitable for use with wxPython >= 4 (a.k.a. Phoenix).

    It uses the same approach to that used in
    ipykernel.eventloops.loop_wx. The wx.MainLoop is executed, and a wx.Timer
    is used to periodically poll the context for input. As soon as input is
    ready, the wx.MainLoop is stopped.
    """
    app = wx.GetApp()
    if app is None:
        return
    if context.input_is_ready():
        return
    assert wx.IsMainThread()
    poll_interval = 100
    timer = wx.Timer()

    def poll(ev):
        if context.input_is_ready():
            timer.Stop()
            app.ExitMainLoop()
    timer.Start(poll_interval)
    timer.Bind(wx.EVT_TIMER, poll)
    if not callable(signal.getsignal(signal.SIGINT)):
        signal.signal(signal.SIGINT, signal.default_int_handler)
    app.SetExitOnFrameDelete(False)
    app.MainLoop()