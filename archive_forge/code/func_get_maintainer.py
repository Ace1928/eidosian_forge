import email.utils
import logging
import os
import re
import socket
from debian.debian_support import Version
def get_maintainer():
    """Get the maintainer information in the same manner as dch.

    This function gets the information about the current user for
    the maintainer field using environment variables of gecos
    information as appropriate.

    It uses the same algorithm as dch to get the information, namely
    DEBEMAIL, DEBFULLNAME, EMAIL, NAME, /etc/mailname and gecos.

    :returns: a tuple of the full name, email pair as strings.
        Either of the pair may be None if that value couldn't
        be determined.
    """
    env = os.environ
    if 'DEBEMAIL' in env:
        match_obj = maintainerre.match(env['DEBEMAIL'])
        if match_obj:
            if 'DEBFULLNAME' not in env:
                env['DEBFULLNAME'] = match_obj.group(1)
            env['DEBEMAIL'] = match_obj.group(2)
    if 'DEBEMAIL' not in env or 'DEBFULLNAME' not in env:
        if 'EMAIL' in env:
            match_obj = maintainerre.match(env['EMAIL'])
            if match_obj:
                if 'DEBFULLNAME' not in env:
                    env['DEBFULLNAME'] = match_obj.group(1)
                env['EMAIL'] = match_obj.group(2)
    maintainer = None
    if 'DEBFULLNAME' in env:
        maintainer = env['DEBFULLNAME']
    elif 'NAME' in env:
        maintainer = env['NAME']
    else:
        try:
            maintainer = re.sub(',.*', '', pwd.getpwuid(os.getuid()).pw_gecos)
        except (KeyError, AttributeError, NameError):
            pass
    email_address = None
    if 'DEBEMAIL' in env:
        email_address = env['DEBEMAIL']
    elif 'EMAIL' in env:
        email_address = env['EMAIL']
    else:
        addr = None
        if os.path.exists('/etc/mailname'):
            with open('/etc/mailname', encoding='UTF-8') as f:
                addr = f.readline().strip()
        if not addr:
            addr = socket.getfqdn()
        if addr:
            try:
                user = pwd.getpwuid(os.getuid()).pw_name
            except (AttributeError, NameError):
                addr = None
            else:
                if not user:
                    addr = None
                else:
                    addr = '%s@%s' % (user, addr)
        if addr:
            email_address = addr
    return (maintainer, email_address)