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
def fetch_keys(proto, username, useragent):
    """
    Call out to a subcommand to handle the specified protocol and username
    """
    if proto == 'lp':
        return fetch_keys_lp(username, useragent)
    if proto == 'gh':
        return fetch_keys_gh(username, useragent)
    die('ssh-import-id protocol handler %s: not found or cannot execute' % proto)