import asyncore
import binascii
import collections
import errno
import functools
import hashlib
import hmac
import math
import os
import pickle
import socket
import struct
import time
import futurist
from oslo_utils import excutils
from taskflow.engines.action_engine import executor as base
from taskflow import logging
from taskflow import task as ta
from taskflow.types import notifier as nt
from taskflow.utils import iter_utils
from taskflow.utils import misc
from taskflow.utils import schema_utils as su
from taskflow.utils import threading_utils
def rebind_task():
    proxy_event_types = set()
    for event_type, listeners in task.notifier.listeners_iter():
        if listeners:
            proxy_event_types.add(event_type)
    if progress_callback is not None:
        proxy_event_types.add(ta.EVENT_UPDATE_PROGRESS)
    if nt.Notifier.ANY in proxy_event_types:
        proxy_event_types = set([nt.Notifier.ANY])
    if proxy_event_types:
        sender = EventSender(channel)
        for event_type in proxy_event_types:
            clone.notifier.register(event_type, sender)
    return bool(proxy_event_types)