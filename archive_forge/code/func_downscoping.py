import os
import pathlib
import subprocess
import shutil
import tempfile
from nox.command import which
import nox
@nox.session(python=PYTHON_VERSIONS_SYNC)
def downscoping(session):
    session.install(*TEST_DEPENDENCIES_SYNC, LIBRARY_DIR, 'google-cloud-storage')
    default(session, 'system_tests_sync/test_downscoping.py', *session.posargs)