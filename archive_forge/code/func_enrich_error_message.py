from __future__ import (absolute_import, division, print_function)
import getpass
import os
import socket
import sys
import time
import uuid
from collections import OrderedDict
from os.path import basename
from ansible.errors import AnsibleError
from ansible.module_utils.six import raise_from
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.plugins.callback import CallbackBase
@staticmethod
def enrich_error_message(result):
    message = result.get('msg', 'failed')
    exception = result.get('exception')
    stderr = result.get('stderr')
    return 'message: "{0}"\nexception: "{1}"\nstderr: "{2}"'.format(message, exception, stderr)