import abc
import collections
import copy
import errno
import functools
import gc
import inspect
import io
import logging
import os
import random
import signal
import sys
import time
import eventlet
from eventlet import event
from eventlet import tpool
from oslo_concurrency import lockutils
from oslo_service._i18n import _
from oslo_service import _options
from oslo_service import eventlet_backdoor
from oslo_service import systemd
from oslo_service import threadgroup
def launch_service(self, service, workers=1):
    """Launch a service with a given number of workers.

       :param service: a service to launch, must be an instance of
              :class:`oslo_service.service.ServiceBase`
       :param workers: a number of processes in which a service
              will be running
        """
    _check_service_base(service)
    wrap = ServiceWrapper(service, workers)
    if hasattr(gc, 'freeze'):
        gc.freeze()
    LOG.info('Starting %d workers', wrap.workers)
    while self.running and len(wrap.children) < wrap.workers:
        self._start_child(wrap)