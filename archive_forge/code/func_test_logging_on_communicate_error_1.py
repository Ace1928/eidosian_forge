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
def test_logging_on_communicate_error_1(self):
    self._test_and_check_logging_communicate_errors(log_errors=processutils.LOG_FINAL_ERROR, attempts=None)