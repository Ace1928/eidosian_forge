from __future__ import with_statement, print_function
import abc
import sys
from optparse import OptionParser
import rsa
import rsa.pkcs1
def read_key(self, filename, keyform):
    """Reads a public or private key."""
    print('Reading %s key from %s' % (self.keyname, filename), file=sys.stderr)
    with open(filename, 'rb') as keyfile:
        keydata = keyfile.read()
    return self.key_class.load_pkcs1(keydata, keyform)