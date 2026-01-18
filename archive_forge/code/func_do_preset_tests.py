import os, sys
import re
import logging; log = logging.getLogger(__name__)
from passlib.utils.compat import print_
def do_preset_tests(name):
    """return list of preset test names"""
    if name == 'django' or name == 'django-hashes':
        do_hash_tests('django_.*_test', 'hex_md5_test')
        if name == 'django':
            print_('passlib.tests.test_ext_django')
    else:
        raise ValueError('unknown name: %r' % name)