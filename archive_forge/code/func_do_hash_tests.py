import os, sys
import re
import logging; log = logging.getLogger(__name__)
from passlib.utils.compat import print_
def do_hash_tests(*args):
    """return list of hash algorithm tests that match regexes"""
    if not args:
        print(TH_PATH)
        return
    suffix = ''
    args = list(args)
    while True:
        if args[0] == '--method':
            suffix = '.' + args[1]
            del args[:2]
        else:
            break
    from passlib.tests import test_handlers
    names = [TH_PATH + ':' + name + suffix for name in dir(test_handlers) if not name.startswith('_') and any((re.match(arg, name) for arg in args))]
    print_('\n'.join(names))
    return not names