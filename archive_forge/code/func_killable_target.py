import sys
from time import monotonic
from greenlet import GreenletExit
from kombu.asynchronous import timer as _timer
from celery import signals
from . import base
def killable_target(*args, **kwargs):
    try:
        return target(*args, **kwargs)
    except GreenletExit:
        return (False, None, None)