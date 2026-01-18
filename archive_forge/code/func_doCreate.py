import os
import sys
from zope.interface import implementer
import pywintypes
import win32api
import win32con
import win32event
import win32file
import win32pipe
import win32process
import win32security
from twisted.internet import _pollingfile, error
from twisted.internet._baseprocess import BaseProcess
from twisted.internet.interfaces import IConsumer, IProcessTransport, IProducer
from twisted.python.win32 import quoteArguments
def doCreate():
    flags = win32con.CREATE_NO_WINDOW
    self.hProcess, self.hThread, self.pid, dwTid = win32process.CreateProcess(command, cmdline, None, None, 1, flags, env, path, StartupInfo)