import contextlib
import errno
import functools
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import weakref
import fasteners
from oslo_config import cfg
from oslo_utils import reflection
from oslo_utils import timeutils
from oslo_concurrency._i18n import _
def synchronized_with_prefix(lock_file_prefix):
    """Partial object generator for the synchronization decorator.

    Redefine @synchronized in each project like so::

        (in nova/utils.py)
        from oslo_concurrency import lockutils

        _prefix = 'nova'
        synchronized = lockutils.synchronized_with_prefix(_prefix)
        lock_cleanup = lockutils.remove_external_lock_file_with_prefix(_prefix)


        (in nova/foo.py)
        from nova import utils

        @utils.synchronized('mylock')
        def bar(self, *args):
           ...

    Eventually clean up with::

        lock_cleanup('mylock')

    :param lock_file_prefix: A string used to provide lock files on disk with a
        meaningful prefix. Will be separated from the lock name with a hyphen,
        which may optionally be included in the lock_file_prefix (e.g.
        ``'nova'`` and ``'nova-'`` are equivalent).
    """
    return functools.partial(synchronized, lock_file_prefix=lock_file_prefix)