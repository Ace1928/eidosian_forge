import asyncio
import datetime
import hashlib
import inspect
import re
import time as mod_time
import warnings
from typing import (
from aioredis.compat import Protocol, TypedDict
from aioredis.connection import (
from aioredis.exceptions import (
from aioredis.lock import Lock
from aioredis.utils import safe_str, str_if_bytes
def handle_message(self, response, ignore_subscribe_messages=False):
    """
        Parses a pub/sub message. If the channel or pattern was subscribed to
        with a message handler, the handler is invoked instead of a parsed
        message being returned.
        """
    message_type = str_if_bytes(response[0])
    if message_type == 'pmessage':
        message = {'type': message_type, 'pattern': response[1], 'channel': response[2], 'data': response[3]}
    elif message_type == 'pong':
        message = {'type': message_type, 'pattern': None, 'channel': None, 'data': response[1]}
    else:
        message = {'type': message_type, 'pattern': None, 'channel': response[1], 'data': response[2]}
    if message_type in self.UNSUBSCRIBE_MESSAGE_TYPES:
        if message_type == 'punsubscribe':
            pattern = response[1]
            if pattern in self.pending_unsubscribe_patterns:
                self.pending_unsubscribe_patterns.remove(pattern)
                self.patterns.pop(pattern, None)
        else:
            channel = response[1]
            if channel in self.pending_unsubscribe_channels:
                self.pending_unsubscribe_channels.remove(channel)
                self.channels.pop(channel, None)
    if message_type in self.PUBLISH_MESSAGE_TYPES:
        if message_type == 'pmessage':
            handler = self.patterns.get(message['pattern'], None)
        else:
            handler = self.channels.get(message['channel'], None)
        if handler:
            handler(message)
            return None
    elif message_type != 'pong':
        if ignore_subscribe_messages or self.ignore_subscribe_messages:
            return None
    return message