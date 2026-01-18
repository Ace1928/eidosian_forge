import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
def getquota(self, root):
    """Get the quota root's resource usage and limits.

        Part of the IMAP4 QUOTA extension defined in rfc2087.

        (typ, [data]) = <instance>.getquota(root)
        """
    typ, dat = self._simple_command('GETQUOTA', root)
    return self._untagged_response(typ, dat, 'QUOTA')