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
def process_IN_Q_OVERFLOW(self, event):
    """
        By default this method only reports warning messages, you can overredide
        it by subclassing ProcessEvent and implement your own
        process_IN_Q_OVERFLOW method. The actions you can take on receiving this
        event is either to update the variable max_queued_events in order to
        handle more simultaneous events or to modify your code in order to
        accomplish a better filtering diminishing the number of raised events.
        Because this method is defined, IN_Q_OVERFLOW will never get
        transmitted as arguments to process_default calls.

        @param event: IN_Q_OVERFLOW event.
        @type event: dict
        """
    log.warning('Event queue overflowed.')