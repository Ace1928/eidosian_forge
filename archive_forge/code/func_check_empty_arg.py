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
def check_empty_arg(arg, arg_descriptor):
    if not arg.strip():
        raise exc.CommandError(_('%(arg)s cannot be empty or only have blank spaces') % {'arg': arg_descriptor})