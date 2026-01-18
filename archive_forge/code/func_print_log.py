import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
def print_log(self):
    self._mesg('last %d IMAP4 interactions:' % len(self._cmd_log))
    i, n = (self._cmd_log_idx, self._cmd_log_len)
    while n:
        try:
            self._mesg(*self._cmd_log[i])
        except:
            pass
        i += 1
        if i >= self._cmd_log_len:
            i = 0
        n -= 1