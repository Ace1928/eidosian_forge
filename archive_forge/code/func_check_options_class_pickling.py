import abc
import bz2
from datetime import date, datetime
from decimal import Decimal
import gc
import gzip
import io
import itertools
import os
import select
import shutil
import signal
import string
import tempfile
import threading
import time
import unittest
import weakref
import pytest
import numpy as np
import pyarrow as pa
from pyarrow.csv import (
from pyarrow.tests import util
def check_options_class_pickling(cls, pickler, **attr_values):
    opts = cls(**attr_values)
    new_opts = pickler.loads(pickler.dumps(opts, protocol=pickler.HIGHEST_PROTOCOL))
    for name, value in attr_values.items():
        assert getattr(new_opts, name) == value