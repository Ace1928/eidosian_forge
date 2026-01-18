import os
import platform
import sys
from functools import partial
import zmq
from packaging.version import Version as V
from traitlets.config.application import Application
@loop_wx.exit
def loop_wx_exit(kernel):
    """Exit the wx loop."""
    import wx
    wx.Exit()