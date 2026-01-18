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
def get_random_filenames(self) -> typing.Sequence[pathlib.Path]:
    filenames = list(self.get_filenames())
    random.shuffle(filenames)
    return filenames