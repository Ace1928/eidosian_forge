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
def get_visible(self):
    """Return a Message representation of visible headers."""
    return Message(self._visible)