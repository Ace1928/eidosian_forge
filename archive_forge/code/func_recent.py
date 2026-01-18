import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
def recent(self):
    """Return most recent 'RECENT' responses if any exist,
        else prompt server for an update using the 'NOOP' command.

        (typ, [data]) = <instance>.recent()

        'data' is None if no new messages,
        else list of RECENT responses, most recent last.
        """
    name = 'RECENT'
    typ, dat = self._untagged_response('OK', [None], name)
    if dat[-1]:
        return (typ, dat)
    typ, dat = self.noop()
    return self._untagged_response(typ, dat, name)