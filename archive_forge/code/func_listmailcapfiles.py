import os
import warnings
import re
def listmailcapfiles():
    """Return a list of all mailcap files found on the system."""
    if 'MAILCAPS' in os.environ:
        pathstr = os.environ['MAILCAPS']
        mailcaps = pathstr.split(os.pathsep)
    else:
        if 'HOME' in os.environ:
            home = os.environ['HOME']
        else:
            home = '.'
        mailcaps = [home + '/.mailcap', '/etc/mailcap', '/usr/etc/mailcap', '/usr/local/etc/mailcap']
    return mailcaps