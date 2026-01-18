import os
import pathlib
import subprocess
import shutil
import tempfile
from nox.command import which
import nox
@nox.session(python=PYTHON_VERSIONS_ASYNC)
def external_accounts(session):
    session.env[ALLOW_PLUGGABLE_ENV] = '1'
    session.install(*TEST_DEPENDENCIES_ASYNC, LIBRARY_DIR, 'google-api-python-client')
    default(session, 'system_tests_sync/test_external_accounts.py', *session.posargs)