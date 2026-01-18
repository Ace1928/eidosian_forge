import os
import signal
import sys
from functools import wraps
import click
from kombu.utils.objects import cached_property
from celery import VERSION_BANNER
from celery.apps.multi import Cluster, MultiParser, NamespacedOptionParser
from celery.bin.base import CeleryCommand, handle_preload_options
from celery.platforms import EX_FAILURE, EX_OK, signals
from celery.utils import term
from celery.utils.text import pluralize
def on_still_waiting_for(self, nodes):
    num_left = len(nodes)
    if num_left:
        self.note(self.colored.blue('> Waiting for {} {} -> {}...'.format(num_left, pluralize(num_left, 'node'), ', '.join((str(node.pid) for node in nodes)))), newline=False)