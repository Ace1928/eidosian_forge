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
def remove_keys(proto, username):
    """
    Remove keys from the output file, if they were inserted by this tool
    """
    comment_string = '# ssh-import-id %s:%s\n' % (proto, username)
    update_lines = []
    removed = []
    for line in read_keyfile():
        if line.endswith(comment_string):
            ssh_fp = key_fingerprint(line.split())
            logging.info('Removed labeled key %s', ssh_fp[:3] + ssh_fp[-1:])
            removed.append(line)
        else:
            update_lines.append(line)
    write_keyfile(update_lines, 'w')
    return removed