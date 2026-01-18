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
def key_fingerprint(fields):
    """
    Get the fingerprint for an SSH public key
    Returns None if not valid key material
    """
    if not fields:
        return None
    if len(fields) < 3:
        return None
    tempfd, tempname = tempfile.mkstemp(prefix='ssh-auth-key-check', suffix='.pub')
    TEMPFILES.append(tempname)
    with os.fdopen(tempfd, 'w') as tempf:
        tempf.write(' '.join(fields))
        tempf.write('\n')
    keygen_proc = subprocess.Popen(['ssh-keygen', '-l', '-f', tempname], stdout=subprocess.PIPE)
    keygen_out, _ = keygen_proc.communicate(None)
    if keygen_proc.returncode:
        return None
    os.unlink(tempname)
    keygen_fields = keygen_out.split()
    if not keygen_fields or len(keygen_fields) < 2:
        return None
    out = []
    for k in keygen_out.split():
        out.append(str(k.decode('utf-8').strip()))
    return out