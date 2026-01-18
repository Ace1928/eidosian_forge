import sys
import threading
import os
import select
import struct
import fcntl
import errno
import termios
import array
import logging
import atexit
from collections import deque
from datetime import datetime, timedelta
import time
import re
import asyncore
import glob
import locale
import subprocess
def coalesce_events(self, coalesce=True):
    """
        Coalescing events. Events are usually processed by batchs, their size
        depend on various factors. Thus, before processing them, events received
        from inotify are aggregated in a fifo queue. If this coalescing
        option is enabled events are filtered based on their unicity, only
        unique events are enqueued, doublons are discarded. An event is unique
        when the combination of its fields (wd, mask, cookie, name) is unique
        among events of a same batch. After a batch of events is processed any
        events is accepted again. By default this option is disabled, you have
        to explictly call this function to turn it on.

        @param coalesce: Optional new coalescing value. True by default.
        @type coalesce: Bool
        """
    self._coalesce = coalesce
    if not coalesce:
        self._eventset.clear()