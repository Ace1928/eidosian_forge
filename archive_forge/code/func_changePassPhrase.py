from __future__ import annotations
import getpass
import os
import platform
import socket
import sys
from collections.abc import Callable
from functools import wraps
from importlib import reload
from typing import Any, Dict, Optional
from twisted.conch.ssh import keys
from twisted.python import failure, filepath, log, usage
def changePassPhrase(options):
    filename = _getKeyOrDefault(options)
    try:
        key = keys.Key.fromFile(filename)
    except keys.EncryptedKeyError:
        if not options.get('pass'):
            options['pass'] = getpass.getpass('Enter old passphrase: ')
        try:
            key = keys.Key.fromFile(filename, passphrase=options['pass'])
        except keys.BadKeyError:
            sys.exit('Could not change passphrase: old passphrase error')
        except keys.EncryptedKeyError as e:
            sys.exit(f'Could not change passphrase: {e}')
    except keys.BadKeyError as e:
        sys.exit(f'Could not change passphrase: {e}')
    except FileNotFoundError:
        sys.exit(f'{filename} could not be opened, please specify a file.')
    if not options.get('newpass'):
        while 1:
            p1 = getpass.getpass('Enter new passphrase (empty for no passphrase): ')
            p2 = getpass.getpass('Enter same passphrase again: ')
            if p1 == p2:
                break
            print('Passphrases do not match.  Try again.')
        options['newpass'] = p1
    if options.get('private-key-subtype') is None:
        options['private-key-subtype'] = _defaultPrivateKeySubtype(key.type())
    try:
        newkeydata = key.toString('openssh', subtype=options['private-key-subtype'], passphrase=options['newpass'])
    except Exception as e:
        sys.exit(f'Could not change passphrase: {e}')
    try:
        keys.Key.fromString(newkeydata, passphrase=options['newpass'])
    except (keys.EncryptedKeyError, keys.BadKeyError) as e:
        sys.exit(f'Could not change passphrase: {e}')
    with open(filename, 'wb') as fd:
        fd.write(newkeydata)
    print('Your identification has been saved with the new passphrase.')