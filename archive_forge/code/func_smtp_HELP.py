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
def smtp_HELP(self, arg):
    if arg:
        extended = ' [SP <mail-parameters>]'
        lc_arg = arg.upper()
        if lc_arg == 'EHLO':
            self.push('250 Syntax: EHLO hostname')
        elif lc_arg == 'HELO':
            self.push('250 Syntax: HELO hostname')
        elif lc_arg == 'MAIL':
            msg = '250 Syntax: MAIL FROM: <address>'
            if self.extended_smtp:
                msg += extended
            self.push(msg)
        elif lc_arg == 'RCPT':
            msg = '250 Syntax: RCPT TO: <address>'
            if self.extended_smtp:
                msg += extended
            self.push(msg)
        elif lc_arg == 'DATA':
            self.push('250 Syntax: DATA')
        elif lc_arg == 'RSET':
            self.push('250 Syntax: RSET')
        elif lc_arg == 'NOOP':
            self.push('250 Syntax: NOOP')
        elif lc_arg == 'QUIT':
            self.push('250 Syntax: QUIT')
        elif lc_arg == 'VRFY':
            self.push('250 Syntax: VRFY <address>')
        else:
            self.push('501 Supported commands: EHLO HELO MAIL RCPT DATA RSET NOOP QUIT VRFY')
    else:
        self.push('250 Supported commands: EHLO HELO MAIL RCPT DATA RSET NOOP QUIT VRFY')