import os
import platform
import shelve
import sys
import weakref
import zlib
from collections import Counter
from kombu.serialization import pickle, pickle_protocol
from kombu.utils.objects import cached_property
from celery import __version__
from celery.exceptions import WorkerShutdown, WorkerTerminate
from celery.utils.collections import LimitedSet
def maybe_shutdown():
    """Shutdown if flags have been set."""
    if should_terminate is not None and should_terminate is not False:
        raise WorkerTerminate(should_terminate)
    elif should_stop is not None and should_stop is not False:
        raise WorkerShutdown(should_stop)