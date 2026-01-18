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
def remove_label(self, label):
    """Remove label from the list of labels on the message."""
    try:
        self._labels.remove(label)
    except ValueError:
        pass