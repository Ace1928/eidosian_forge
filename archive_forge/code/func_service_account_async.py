import os
import pathlib
import subprocess
import shutil
import tempfile
from nox.command import which
import nox
@nox.session(python=PYTHON_VERSIONS_ASYNC)
def service_account_async(session):
    session.install(*TEST_DEPENDENCIES_SYNC + TEST_DEPENDENCIES_ASYNC)
    session.install(LIBRARY_DIR)
    default(session, 'system_tests_async/test_service_account.py', *session.posargs)