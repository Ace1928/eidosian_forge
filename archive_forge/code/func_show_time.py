import contextlib
import logging
import os
import random
import sys
import time
from oslo_utils import reflection
from taskflow import engines
from taskflow.listeners import printing
from taskflow.patterns import unordered_flow as uf
from taskflow import task
@contextlib.contextmanager
def show_time(name):
    start = time.time()
    yield
    end = time.time()
    print(' -- %s took %0.3f seconds' % (name, end - start))