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
def check_options_class(cls, **attr_values):
    """
    Check setting and getting attributes of an *Options class.
    """
    opts = cls()
    for name, values in attr_values.items():
        assert getattr(opts, name) == values[0], 'incorrect default value for ' + name
        for v in values:
            setattr(opts, name, v)
            assert getattr(opts, name) == v, 'failed setting value'
    with pytest.raises(AttributeError):
        opts.zzz_non_existent = True
    non_defaults = {name: values[1] for name, values in attr_values.items()}
    opts = cls(**non_defaults)
    for name, value in non_defaults.items():
        assert getattr(opts, name) == value