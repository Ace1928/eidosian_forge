import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
def myrights(self, mailbox):
    """Show my ACLs for a mailbox (i.e. the rights that I have on mailbox).

        (typ, [data]) = <instance>.myrights(mailbox)
        """
    typ, dat = self._simple_command('MYRIGHTS', mailbox)
    return self._untagged_response(typ, dat, 'MYRIGHTS')