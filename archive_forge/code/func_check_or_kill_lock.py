import _thread
import json
import logging
import random
import time
import typing
from redis import client
from . import exceptions, utils
def check_or_kill_lock(self, connection, timeout):
    response_channel = f'{self.channel}-{random.random()}'
    pubsub = connection.pubsub()
    pubsub.subscribe(response_channel)
    connection.publish(self.channel, json.dumps(dict(response_channel=response_channel, message='ping')))
    check_interval = min(self.thread_sleep_time, timeout / 10)
    for _ in self._timeout_generator(timeout, check_interval):
        if pubsub.get_message(timeout=check_interval):
            pubsub.close()
            return True
    for client_ in connection.client_list('pubsub'):
        if client_.get('name') == self.client_name:
            logger.warning('Killing unavailable redis client: %r', client_)
            connection.client_kill_filter(client_.get('id'))
    return None