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
def set_visible(self, visible):
    """Set the Message representation of visible headers."""
    self._visible = Message(visible)