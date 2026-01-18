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
def positive_non_zero_float(text):
    if text is None:
        return None
    try:
        value = float(text)
    except ValueError:
        msg = '%s must be a float' % text
        raise argparse.ArgumentTypeError(msg)
    if value <= 0:
        msg = '%s must be greater than 0' % text
        raise argparse.ArgumentTypeError(msg)
    return value