from __future__ import with_statement
from contextlib import contextmanager
import collections.abc
import logging
import warnings
import numbers
from html.entities import name2codepoint as n2cp
import pickle as _pickle
import re
import unicodedata
import os
import random
import itertools
import tempfile
from functools import wraps
import multiprocessing
import shutil
import sys
import subprocess
import inspect
import heapq
from copy import deepcopy
from datetime import datetime
import platform
import types
import numpy as np
import scipy.sparse
from smart_open import open
from gensim import __version__ as gensim_version
def add_lifecycle_event(self, event_name, log_level=logging.INFO, **event):
    """
        Append an event into the `lifecycle_events` attribute of this object, and also
        optionally log the event at `log_level`.

        Events are important moments during the object's life, such as "model created",
        "model saved", "model loaded", etc.

        The `lifecycle_events` attribute is persisted across object's :meth:`~gensim.utils.SaveLoad.save`
        and :meth:`~gensim.utils.SaveLoad.load` operations. It has no impact on the use of the model,
        but is useful during debugging and support.

        Set `self.lifecycle_events = None` to disable this behaviour. Calls to `add_lifecycle_event()`
        will not record events into `self.lifecycle_events` then.

        Parameters
        ----------
        event_name : str
            Name of the event. Can be any label, e.g. "created", "stored" etc.
        event : dict
            Key-value mapping to append to `self.lifecycle_events`. Should be JSON-serializable, so keep it simple.
            Can be empty.

            This method will automatically add the following key-values to `event`, so you don't have to specify them:

            - `datetime`: the current date & time
            - `gensim`: the current Gensim version
            - `python`: the current Python version
            - `platform`: the current platform
            - `event`: the name of this event
        log_level : int
            Also log the complete event dict, at the specified log level. Set to False to not log at all.

        """
    event_dict = deepcopy(event)
    event_dict['datetime'] = datetime.now().isoformat()
    event_dict['gensim'] = gensim_version
    event_dict['python'] = sys.version
    event_dict['platform'] = platform.platform()
    event_dict['event'] = event_name
    if not hasattr(self, 'lifecycle_events'):
        logger.debug('starting a new internal lifecycle event log for %s', self.__class__.__name__)
        self.lifecycle_events = []
    if log_level:
        logger.log(log_level, '%s lifecycle event %s', self.__class__.__name__, event_dict)
    if self.lifecycle_events is not None:
        self.lifecycle_events.append(event_dict)