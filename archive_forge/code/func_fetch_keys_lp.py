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
def fetch_keys_lp(lpid, useragent):
    conf_file = '/etc/ssh/ssh_import_id'
    try:
        url = os.getenv('URL', None)
        if url is None and os.path.exists(conf_file):
            try:
                contents = open(conf_file).read()
            except OSError:
                raise Exception('Failed to read %s' % conf_file)
            try:
                conf = json.loads(contents)
            except JSONDecodeError:
                raise Exception('File %s did not have valid JSON.' % conf_file)
            url = conf.get('URL', None) % quote_plus(lpid)
        elif url is not None:
            url = url % quote_plus(lpid)
        if url is None:
            url = 'https://launchpad.net/~%s/+sshkeys' % quote_plus(lpid)
        headers = {'User-Agent': user_agent(useragent)}
        try:
            with urlopen(Request(url, headers=headers), timeout=DEFAULT_TIMEOUT) as response:
                keys = response.read().decode('utf-8')
        except urllib.error.HTTPError as e:
            msg = 'Requesting Launchpad keys failed.'
            if e.code == 404:
                msg = 'Launchpad user not found.'
            die(msg + ' status_code=%d user=%s' % (e.code, lpid))
    except Exception as e:
        die(str(e))
    return keys