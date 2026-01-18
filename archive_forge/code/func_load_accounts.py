import os
import re
import logging
import collections
import pyzor.account
def load_accounts(filepath):
    """Layout of file is: host : port : username : salt,key"""
    accounts = {}
    log = logging.getLogger('pyzor')
    if os.path.exists(filepath):
        accountsf = open(filepath)
        for lineno, orig_line in enumerate(accountsf):
            line = orig_line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                host, port, username, key = [x.strip() for x in line.split(':')]
            except ValueError:
                log.warn('account file: invalid line %d: wrong number of parts', lineno)
                continue
            try:
                port = int(port)
            except ValueError as ex:
                log.warn('account file: invalid line %d: %s', lineno, ex)
                continue
            address = (host, port)
            try:
                salt, key = pyzor.account.key_from_hexstr(key)
            except ValueError as ex:
                log.warn('account file: invalid line %d: %s', lineno, ex)
                continue
            if not salt and (not key):
                log.warn("account file: invalid line %d: keystuff can't be all None's", lineno)
                continue
            accounts[address] = pyzor.account.Account(username, salt, key)
        accountsf.close()
    else:
        log.warn('No accounts are setup.  All commands will be executed by the anonymous user.')
    return accounts