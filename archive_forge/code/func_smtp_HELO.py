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
def smtp_HELO(self, arg):
    if not arg:
        self.push('501 Syntax: HELO hostname')
        return
    if self.seen_greeting:
        self.push('503 Duplicate HELO/EHLO')
        return
    self._set_rset_state()
    self.seen_greeting = arg
    self.push('250 %s' % self.fqdn)