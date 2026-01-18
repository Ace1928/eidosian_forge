import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def process_cmd_line():
    """Set up and parse command line options"""
    parser = create_options_parser()
    options = parser.parse_args()
    return (options, parser)