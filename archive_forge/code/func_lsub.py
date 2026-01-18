import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
def lsub(self, directory='""', pattern='*'):
    """List 'subscribed' mailbox names in directory matching pattern.

        (typ, [data, ...]) = <instance>.lsub(directory='""', pattern='*')

        'data' are tuples of message part envelope and data.
        """
    name = 'LSUB'
    typ, dat = self._simple_command(name, directory, pattern)
    return self._untagged_response(typ, dat, name)