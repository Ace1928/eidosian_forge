import sys
import signal
import time
from timeit import default_timer as clock
import wx
def check_stdin(self):
    if self.input_is_ready():
        self.timer.Stop()
        self.evtloop.Exit()