from __future__ import (absolute_import, division, print_function)
import os.path
import socket
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
from base64 import b64encode
from netrc import netrc
from os import environ
from time import sleep
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.module_utils.urls import urllib_error
from stat import S_IRUSR, S_IWUSR
from tempfile import gettempdir, NamedTemporaryFile
import yaml
from ansible.module_utils.urls import open_url
from ansible.utils.display import Display
def parameters_wrapper(target):

    def decorator(*args, **kwargs):
        retry_count = 0
        while True:
            retry_count += 1
            try:
                return_value = target(*args, **kwargs)
                return return_value
            except urllib_error.HTTPError as e:
                if retry_count >= retries:
                    raise e
                display.v('Error encountered. Retrying..')
            except socket.timeout:
                if retry_count >= retries:
                    raise e
                display.v('Socket timeout encountered. Retrying..')
            sleep(retry_interval)
    return decorator