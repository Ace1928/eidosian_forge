import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
def setacl(self, mailbox, who, what):
    """Set a mailbox acl.

        (typ, [data]) = <instance>.setacl(mailbox, who, what)
        """
    return self._simple_command('SETACL', mailbox, who, what)