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
def append_event(self, event):
    """
        Append a raw event to the event queue.

        @param event: An event.
        @type event: _RawEvent instance.
        """
    self._eventq.append(event)