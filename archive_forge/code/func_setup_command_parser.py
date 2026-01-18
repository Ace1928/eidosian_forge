import os
import os.path
import sys
import glob
import textwrap
import logging
import socket
import subprocess
import pyomo.common
from pyomo.common.collections import Bunch
import pyomo.scripting.pyomo_parser
def setup_command_parser(parser):
    parser.add_argument('--list', dest='summary', action='store_true', default=False, help='List the commands that are installed with Pyomo')
    parser.add_argument('command', nargs='*', help='The command and command-line options')