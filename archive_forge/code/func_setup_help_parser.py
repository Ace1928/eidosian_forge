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
def setup_help_parser(parser):
    parser.add_argument('--asciidoc', dest='asciidoc', action='store_true', default=False, help="Generate output that is compatible with asciidoc's markup language")
    parser.add_argument('-c', '--commands', dest='commands', action='store_true', default=False, help='List the commands that are installed with Pyomo')
    parser.add_argument('--components', dest='components', action='store_true', default=False, help="List the components that are available in Pyomo's modeling environment")
    parser.add_argument('-d', '--data-managers', dest='datamanager', action='store_true', default=False, help='Print a summary of the data managers in Pyomo')
    parser.add_argument('-i', '--info', dest='environment', action='store_true', default=False, help='Summarize the environment and Python installation')
    parser.add_argument('-s', '--solvers', dest='solvers', action='store_true', default=False, help='Summarize the available solvers and solver interfaces')
    parser.add_argument('-t', '--transformations', dest='transformations', action='store_true', default=False, help='List the available model transformations')
    parser.add_argument('-w', '--writers', dest='writers', action='store_true', default=False, help='List the available problem writers')
    return parser