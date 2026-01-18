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
def test_core_size(self):
    size = self.soft_limit(resource.RLIMIT_CORE, 1, 1024)
    prlimit = processutils.ProcessLimits(core_file_size=size)
    self.check_limit(prlimit, 'RLIMIT_CORE', prlimit.core_file_size)