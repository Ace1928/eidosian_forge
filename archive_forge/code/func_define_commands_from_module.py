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
def define_commands_from_module(subparsers, command_module, cmd_mapper):
    """Add *do_* methods in a module and add as commands into a subparsers."""
    for method_name in (a for a in dir(command_module) if a.startswith('do_')):
        command = method_name[3:].replace('_', '-')
        callback = getattr(command_module, method_name)
        define_command(subparsers, command, callback, cmd_mapper)