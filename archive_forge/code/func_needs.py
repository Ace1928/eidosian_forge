import optparse
import os
import sys
from subprocess import PIPE, CalledProcessError, Popen
from breezy import osutils
from breezy.tests import ssl_certs
def needs(request, *paths):
    """Errors out if the specified path does not exists"""
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        error('{} needs: {}'.format(request, ','.join(missing)))