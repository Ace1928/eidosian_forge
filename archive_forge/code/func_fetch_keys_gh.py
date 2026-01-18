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
def fetch_keys_gh(ghid, useragent):
    x_ratelimit_remaining = 'x-ratelimit-remaining'
    help_url = 'https://developer.github.com/v3/#rate-limiting'
    keys = ''
    try:
        url = 'https://api.github.com/users/%s/keys' % quote_plus(ghid)
        headers = {'User-Agent': user_agent(useragent)}
        try:
            with urlopen(Request(url, headers=headers), timeout=DEFAULT_TIMEOUT) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as e:
            msg = 'Requesting GitHub keys failed.'
            if e.code == 404:
                msg = 'Username "%s" not found at GitHub API.' % ghid
            elif e.hdrs.get(x_ratelimit_remaining) == '0':
                msg = 'GitHub REST API rate-limited this IP address. See %s .' % help_url
            die(msg + ' status_code=%d user=%s' % (e.code, ghid))
        for keyobj in data:
            keys += '%s %s@github/%s\n' % (keyobj['key'], ghid, keyobj['id'])
    except Exception as e:
        die(str(e))
    return keys