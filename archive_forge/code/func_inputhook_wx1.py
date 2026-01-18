import sys
import signal
import time
from timeit import default_timer as clock
import wx
@ignore_keyboardinterrupts
def inputhook_wx1(context):
    """Run the wx event loop by processing pending events only.

    This approach seems to work, but its performance is not great as it
    relies on having PyOS_InputHook called regularly.
    """
    app = wx.GetApp()
    if app is not None:
        assert wx.Thread_IsMain()
        evtloop = wx.EventLoop()
        ea = wx.EventLoopActivator(evtloop)
        while evtloop.Pending():
            evtloop.Dispatch()
        app.ProcessIdle()
        del ea
    return 0