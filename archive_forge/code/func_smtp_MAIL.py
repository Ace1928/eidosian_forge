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
def smtp_MAIL(self, arg):
    if not self.seen_greeting:
        self.push('503 Error: send HELO first')
        return
    print('===> MAIL', arg, file=DEBUGSTREAM)
    syntaxerr = '501 Syntax: MAIL FROM: <address>'
    if self.extended_smtp:
        syntaxerr += ' [SP <mail-parameters>]'
    if arg is None:
        self.push(syntaxerr)
        return
    arg = self._strip_command_keyword('FROM:', arg)
    address, params = self._getaddr(arg)
    if not address:
        self.push(syntaxerr)
        return
    if not self.extended_smtp and params:
        self.push(syntaxerr)
        return
    if self.mailfrom:
        self.push('503 Error: nested MAIL command')
        return
    self.mail_options = params.upper().split()
    params = self._getparams(self.mail_options)
    if params is None:
        self.push(syntaxerr)
        return
    if not self._decode_data:
        body = params.pop('BODY', '7BIT')
        if body not in ['7BIT', '8BITMIME']:
            self.push('501 Error: BODY can only be one of 7BIT, 8BITMIME')
            return
    if self.enable_SMTPUTF8:
        smtputf8 = params.pop('SMTPUTF8', False)
        if smtputf8 is True:
            self.require_SMTPUTF8 = True
        elif smtputf8 is not False:
            self.push('501 Error: SMTPUTF8 takes no arguments')
            return
    size = params.pop('SIZE', None)
    if size:
        if not size.isdigit():
            self.push(syntaxerr)
            return
        elif self.data_size_limit and int(size) > self.data_size_limit:
            self.push('552 Error: message size exceeds fixed maximum message size')
            return
    if len(params.keys()) > 0:
        self.push('555 MAIL FROM parameters not recognized or not implemented')
        return
    self.mailfrom = address
    print('sender:', self.mailfrom, file=DEBUGSTREAM)
    self.push('250 OK')