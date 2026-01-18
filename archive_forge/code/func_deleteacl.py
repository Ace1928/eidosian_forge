import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
def deleteacl(self, mailbox, who):
    """Delete the ACLs (remove any rights) set for who on mailbox.

        (typ, [data]) = <instance>.deleteacl(mailbox, who)
        """
    return self._simple_command('DELETEACL', mailbox, who)