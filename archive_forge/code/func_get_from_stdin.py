import argparse
import base64
import contextlib
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from oslo_utils import strutils
import yaml
from ironicclient.common.i18n import _
from ironicclient import exc
def get_from_stdin(info_desc):
    """Read information from stdin.

    :param info_desc: A string description of the desired information
    :raises: InvalidAttribute if there was a problem reading from stdin
    :returns: the string that was read from stdin
    """
    try:
        info = sys.stdin.read().strip()
    except Exception as e:
        err = _('Cannot get %(desc)s from standard input. Error: %(err)s')
        raise exc.InvalidAttribute(err % {'desc': info_desc, 'err': e})
    return info