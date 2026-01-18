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
def smtp_RCPT(self, arg):
    if not self.seen_greeting:
        self.push('503 Error: send HELO first')
        return
    print('===> RCPT', arg, file=DEBUGSTREAM)
    if not self.mailfrom:
        self.push('503 Error: need MAIL command')
        return
    syntaxerr = '501 Syntax: RCPT TO: <address>'
    if self.extended_smtp:
        syntaxerr += ' [SP <mail-parameters>]'
    if arg is None:
        self.push(syntaxerr)
        return
    arg = self._strip_command_keyword('TO:', arg)
    address, params = self._getaddr(arg)
    if not address:
        self.push(syntaxerr)
        return
    if not self.extended_smtp and params:
        self.push(syntaxerr)
        return
    self.rcpt_options = params.upper().split()
    params = self._getparams(self.rcpt_options)
    if params is None:
        self.push(syntaxerr)
        return
    if len(params.keys()) > 0:
        self.push('555 RCPT TO parameters not recognized or not implemented')
        return
    self.rcpttos.append(address)
    print('recips:', self.rcpttos, file=DEBUGSTREAM)
    self.push('250 OK')