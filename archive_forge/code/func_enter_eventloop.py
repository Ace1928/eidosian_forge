from __future__ import annotations
import asyncio
import concurrent.futures
import inspect
import itertools
import logging
import os
import socket
import sys
import threading
import time
import typing as t
import uuid
import warnings
from datetime import datetime
from functools import partial
from signal import SIGINT, SIGTERM, Signals, default_int_handler, signal
from .control import CONTROL_THREAD_NAME
import psutil
import zmq
from IPython.core.error import StdinNotImplementedError
from jupyter_client.session import Session
from tornado import ioloop
from tornado.queues import Queue, QueueEmpty
from traitlets.config.configurable import SingletonConfigurable
from traitlets.traitlets import (
from zmq.eventloop.zmqstream import ZMQStream
from ipykernel.jsonutil import json_clean
from ._version import kernel_protocol_version
from .iostream import OutStream
def enter_eventloop(self):
    """enter eventloop"""
    self.log.info('Entering eventloop %s', self.eventloop)
    eventloop = self.eventloop
    if eventloop is None:
        self.log.info('Exiting as there is no eventloop')
        return

    async def advance_eventloop():
        if self.eventloop is not eventloop:
            self.log.info('exiting eventloop %s', eventloop)
            return
        if self.msg_queue.qsize():
            self.log.debug('Delaying eventloop due to waiting messages')
            schedule_next()
            return
        self.log.debug('Advancing eventloop %s', eventloop)
        try:
            eventloop(self)
        except KeyboardInterrupt:
            self.log.error('KeyboardInterrupt caught in kernel')
        if self.eventloop is eventloop:
            schedule_next()

    def schedule_next():
        """Schedule the next advance of the eventloop"""
        self.log.debug('Scheduling eventloop advance')
        self.io_loop.call_later(0.001, partial(self.schedule_dispatch, advance_eventloop))
    schedule_next()