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
def key_list(keyfile_lines):
    """
    Return a list of uniquely identified keys
    """
    keys = []
    for line in keyfile_lines:
        ssh_fp = key_fingerprint(line.split())
        if ssh_fp:
            keys.append(fp_tuple(ssh_fp))
    logging.debug('Already have SSH public keys: [%s]', ' '.join(keys))
    return keys