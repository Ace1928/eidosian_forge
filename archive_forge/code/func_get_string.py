import os
import time
import calendar
import socket
import errno
import copy
import warnings
import email
import email.message
import email.generator
import io
import contextlib
from types import GenericAlias
def get_string(self, key, from_=False):
    """Return a string representation or raise a KeyError."""
    return email.message_from_bytes(self.get_bytes(key, from_)).as_string(unixfrom=from_)