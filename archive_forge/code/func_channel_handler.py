import _thread
import json
import logging
import random
import time
import typing
from redis import client
from . import exceptions, utils
def channel_handler(self, message):
    if message.get('type') != 'message':
        return
    try:
        data = json.loads(message.get('data'))
    except TypeError:
        logger.debug('TypeError while parsing: %r', message)
        return
    assert self.connection is not None
    self.connection.publish(data['response_channel'], str(time.time()))