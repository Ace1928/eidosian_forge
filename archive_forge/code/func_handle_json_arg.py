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
def handle_json_arg(json_arg, info_desc):
    """Read a JSON argument from stdin, file or string.

    :param json_arg: May be a file name containing the JSON, a JSON string, or
        '-' indicating that the argument should be read from standard input.
    :param info_desc: A string description of the desired information
    :returns: A list or dictionary parsed from JSON.
    :raises: InvalidAttribute if the argument cannot be parsed.
    """
    if json_arg == '-':
        json_arg = get_from_stdin(info_desc)
    if json_arg:
        json_arg = handle_json_or_file_arg(json_arg)
    return json_arg