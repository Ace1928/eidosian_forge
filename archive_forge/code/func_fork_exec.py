import errno
import functools
import http.client
import http.server
import io
import os
import shlex
import shutil
import signal
import socket
import subprocess
import threading
import time
from unittest import mock
from alembic import command as alembic_command
import fixtures
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_config import fixture as cfg_fixture
from oslo_log.fixture import logging_error as log_fixture
from oslo_log import log
from oslo_utils import timeutils
from oslo_utils import units
import testtools
import webob
from glance.api.v2 import cached_images
from glance.common import config
from glance.common import exception
from glance.common import property_utils
from glance.common import utils
from glance.common import wsgi
from glance import context
from glance.db.sqlalchemy import alembic_migrations
from glance.db.sqlalchemy import api as db_api
from glance.tests.unit import fixtures as glance_fixtures
def fork_exec(cmd, exec_env=None, logfile=None, pass_fds=None):
    """
    Execute a command using fork/exec.

    This is needed for programs system executions that need path
    searching but cannot have a shell as their parent process, for
    example: glance-api.  When glance-api starts it sets itself as
    the parent process for its own process group.  Thus the pid that
    a Popen process would have is not the right pid to use for killing
    the process group.  This patch gives the test env direct access
    to the actual pid.

    :param cmd: Command to execute as an array of arguments.
    :param exec_env: A dictionary representing the environment with
                     which to run the command.
    :param logfile: A path to a file which will hold the stdout/err of
                   the child process.
    :param pass_fds: Sequence of file descriptors passed to the child.
    """
    env = os.environ.copy()
    if exec_env is not None:
        for env_name, env_val in exec_env.items():
            if callable(env_val):
                env[env_name] = env_val(env.get(env_name))
            else:
                env[env_name] = env_val
    pid = os.fork()
    if pid == 0:
        if logfile:
            fds = [1, 2]
            with open(logfile, 'r+b') as fptr:
                for desc in fds:
                    try:
                        os.dup2(fptr.fileno(), desc)
                    except OSError:
                        pass
        if pass_fds:
            for fd in pass_fds:
                os.set_inheritable(fd, True)
        args = shlex.split(cmd)
        os.execvpe(args[0], args, env)
    else:
        return pid