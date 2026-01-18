import base64
import getpass
import os
import signal
import struct
import sys
import tkinter as Tkinter
import tkinter.filedialog as tkFileDialog
import tkinter.messagebox as tkMessageBox
from typing import List, Tuple
from twisted.conch import error
from twisted.conch.client.default import isInKnownHosts
from twisted.conch.ssh import (
from twisted.conch.ui import tkvt100
from twisted.internet import defer, protocol, reactor, tksupport
from twisted.python import log, usage
def removeForward(self):
    cur = self.forwards.curselection()
    if cur:
        self.forwards.remove(cur[0])