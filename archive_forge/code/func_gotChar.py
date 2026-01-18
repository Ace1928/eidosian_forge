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
def gotChar(ch, resp=resp):
    if not ch:
        return
    if ch == '\x03':
        reactor.stop()
    if ch == '\r':
        frame.write('\r\n')
        stresp = ''.join(resp)
        del resp
        frame.callback = None
        d.callback(stresp)
        return
    elif 32 <= ord(ch) < 127:
        resp.append(ch)
        if echo:
            frame.write(ch)
    elif ord(ch) == 8 and resp:
        if echo:
            frame.write('\x08 \x08')
        resp.pop()