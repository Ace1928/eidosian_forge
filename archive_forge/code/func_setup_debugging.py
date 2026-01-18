import argparse
import logging
import os
import sys
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
from magnumclient.common import cliutils
from magnumclient import exceptions as exc
from magnumclient.i18n import _
from magnumclient.v1 import client as client_v1
from magnumclient.v1 import shell as shell_v1
from magnumclient import version
def setup_debugging(self, debug):
    if debug:
        streamformat = '%(levelname)s (%(module)s:%(lineno)d) %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=streamformat)
    else:
        streamformat = '%(levelname)s %(message)s'
        logging.basicConfig(level=logging.CRITICAL, format=streamformat)