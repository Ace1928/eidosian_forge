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
def getIdentityFile(self):
    r = tkFileDialog.askopenfilename()
    if r:
        self.identity.delete(0, Tkinter.END)
        self.identity.insert(Tkinter.END, r)