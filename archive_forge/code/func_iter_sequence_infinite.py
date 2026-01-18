from abc import abstractmethod
from contextlib import closing
import functools
import hashlib
import multiprocessing
import multiprocessing.dummy
import os
import queue
import random
import shutil
import sys  # pylint: disable=unused-import
import tarfile
import threading
import time
import typing
import urllib
import weakref
import zipfile
import numpy as np
from tensorflow.python.framework import tensor
from six.moves.urllib.request import urlopen
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils.generic_utils import Progbar
from tensorflow.python.keras.utils.io_utils import path_to_string
def iter_sequence_infinite(seq):
    """Iterates indefinitely over a Sequence.

  Args:
    seq: `Sequence` instance.

  Yields:
    Batches of data from the `Sequence`.
  """
    while True:
        for item in seq:
            yield item