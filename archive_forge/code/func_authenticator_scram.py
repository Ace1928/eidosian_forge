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
def authenticator_scram(self):
    client_first = self.first_message().encode('utf-8')
    server_first = (yield (client_first, True))
    self.process_server_first_message(server_first.decode('utf-8'))
    client_final = self.final_message().encode('utf-8')
    server_final = (yield (client_final, True))
    self.process_server_final_message(server_final.decode('utf-8'))