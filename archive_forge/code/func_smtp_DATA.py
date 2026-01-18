import sys
import os
import errno
import getopt
import time
import socket
import collections
from warnings import _deprecated, warn
from email._header_value_parser import get_addr_spec, get_angle_addr
import asyncore
import asynchat
def smtp_DATA(self, arg):
    if not self.seen_greeting:
        self.push('503 Error: send HELO first')
        return
    if not self.rcpttos:
        self.push('503 Error: need RCPT command')
        return
    if arg:
        self.push('501 Syntax: DATA')
        return
    self.smtp_state = self.DATA
    self.set_terminator(b'\r\n.\r\n')
    self.push('354 End data with <CR><LF>.<CR><LF>')