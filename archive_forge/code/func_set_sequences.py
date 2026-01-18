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
def set_sequences(self, sequences):
    """Set the list of sequences that include the message."""
    self._sequences = list(sequences)