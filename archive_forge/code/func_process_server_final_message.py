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
def process_server_final_message(self, server_final):
    params = dict((pair.split('=', 1) for pair in server_final.split(',')))
    if self._server_signature != base64.b64decode(params['v'].encode('utf-8')):
        raise ValueError('Server sent wrong signature!')