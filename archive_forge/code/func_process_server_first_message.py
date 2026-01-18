import asyncio
import collections
import base64
import functools
import hashlib
import hmac
import logging
import random
import socket
import struct
import sys
import time
import traceback
import uuid
import warnings
import weakref
import async_timeout
import aiokafka.errors as Errors
from aiokafka.abc import AbstractTokenProvider
from aiokafka.protocol.api import RequestHeader
from aiokafka.protocol.admin import (
from aiokafka.protocol.commit import (
from aiokafka.util import create_future, create_task, get_running_loop, wait_for
def process_server_first_message(self, server_first):
    self._auth_message += ',' + server_first
    params = dict((pair.split('=', 1) for pair in server_first.split(',')))
    server_nonce = params['r']
    if not server_nonce.startswith(self._nonce):
        raise ValueError('Server nonce, did not start with client nonce!')
    self._nonce = server_nonce
    self._auth_message += ',c=biws,r=' + self._nonce
    salt = base64.b64decode(params['s'].encode('utf-8'))
    iterations = int(params['i'])
    self.create_salted_password(salt, iterations)
    self._client_key = self.hmac(self._salted_password, b'Client Key')
    self._stored_key = self._hashfunc(self._client_key).digest()
    self._client_signature = self.hmac(self._stored_key, self._auth_message.encode('utf-8'))
    self._client_proof = ScramAuthenticator._xor_bytes(self._client_key, self._client_signature)
    self._server_key = self.hmac(self._salted_password, b'Server Key')
    self._server_signature = self.hmac(self._server_key, self._auth_message.encode('utf-8'))