import errno
import io
import logging
import multiprocessing
import os
import pickle
import resource
import socket
import stat
import subprocess
import sys
import tempfile
import time
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_concurrency import processutils
def soft_limit(self, res, substract, default_limit):
    soft_limit, hard_limit = resource.getrlimit(res)
    if soft_limit <= 0:
        soft_limit = default_limit
    else:
        soft_limit -= substract
    return soft_limit