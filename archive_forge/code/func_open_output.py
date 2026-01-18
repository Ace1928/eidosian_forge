import argparse
import getpass
import json
import logging
import os
import subprocess
import sys
import tempfile
import urllib.error
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import distro
from .version import VERSION
def open_output(name, mode='a+'):
    """
    Open output for writing, supporting either stdout or a filename
    """
    if name == '-':
        return False
    return open(name, mode)