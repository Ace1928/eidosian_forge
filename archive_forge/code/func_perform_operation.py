from __future__ import with_statement, print_function
import abc
import sys
from optparse import OptionParser
import rsa
import rsa.pkcs1
def perform_operation(self, indata, pub_key, cli_args):
    """Verifies files."""
    signature_file = cli_args[1]
    with open(signature_file, 'rb') as sigfile:
        signature = sigfile.read()
    try:
        rsa.verify(indata, signature, pub_key)
    except rsa.VerificationError:
        raise SystemExit('Verification failed.')
    print('Verification OK', file=sys.stderr)