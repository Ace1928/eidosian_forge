import copy
import functools
import getpass
import logging
import os
import time
import warnings
from cliff import columns as cliff_columns
from oslo_utils import importutils
from osc_lib import exceptions
from osc_lib.i18n import _
def read_blob_file_contents(blob_file):
    try:
        with open(blob_file) as file:
            blob = file.read().strip()
        return blob
    except IOError:
        msg = _('Error occurred trying to read from file %s')
        raise exceptions.CommandError(msg % blob_file)