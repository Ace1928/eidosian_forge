import logging
import os
import platform
import socket
import string
from base64 import b64encode
from urllib import parse
import certifi
import urllib3
from selenium import __version__
from . import utils
from .command import Command
from .errorhandler import ErrorCode
@classmethod
def get_remote_connection_headers(cls, parsed_url, keep_alive=False):
    """Get headers for remote request.

        :Args:
         - parsed_url - The parsed url
         - keep_alive (Boolean) - Is this a keep-alive connection (default: False)
        """
    system = platform.system().lower()
    if system == 'darwin':
        system = 'mac'
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json;charset=UTF-8', 'User-Agent': f'selenium/{__version__} (python {system})'}
    if parsed_url.username:
        base64string = b64encode(f'{parsed_url.username}:{parsed_url.password}'.encode())
        headers.update({'Authorization': f'Basic {base64string.decode()}'})
    if keep_alive:
        headers.update({'Connection': 'keep-alive'})
    return headers