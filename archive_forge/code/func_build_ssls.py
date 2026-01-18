import optparse
import os
import sys
from subprocess import PIPE, CalledProcessError, Popen
from breezy import osutils
from breezy.tests import ssl_certs
def build_ssls(name, options, builders):
    if options is not None:
        for item in options:
            builder = builders.get(item, None)
            if builder is None:
                error('{} is not a known {}'.format(item, name))
            builder()