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
def fake_execute_raises(*cmd, **kwargs):
    raise processutils.ProcessExecutionError(exit_code=42, stdout='stdout', stderr='stderr', cmd=['this', 'is', 'a', 'command'])