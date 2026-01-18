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
def test_memory_locked(self):
    max_memory = self.memory_limit(resource.RLIMIT_MEMLOCK)
    prlimit = processutils.ProcessLimits(memory_locked=max_memory)
    self.check_limit(prlimit, 'RLIMIT_MEMLOCK', max_memory)