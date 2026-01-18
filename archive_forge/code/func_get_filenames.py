import abc
import atexit
import contextlib
import logging
import os
import pathlib
import random
import tempfile
import time
import typing
import warnings
from . import constants, exceptions, portalocker
def get_filenames(self) -> typing.Sequence[pathlib.Path]:
    return [self.get_filename(n) for n in range(self.maximum)]