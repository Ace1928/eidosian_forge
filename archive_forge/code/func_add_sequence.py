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
def add_sequence(self, sequence):
    """Add sequence to list of sequences including the message."""
    if isinstance(sequence, str):
        if not sequence in self._sequences:
            self._sequences.append(sequence)
    else:
        raise TypeError('sequence type must be str: %s' % type(sequence))