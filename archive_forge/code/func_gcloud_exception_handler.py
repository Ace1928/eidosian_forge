from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import os
import sys
@contextlib.contextmanager
def gcloud_exception_handler():
    """Handles exceptions from gcloud to provide a helpful message."""
    try:
        yield
    except Exception:
        python_version = sys.version_info[:2]
        if python_version < MIN_SUPPORTED_PY3_VERSION or python_version > MAX_SUPPORTED_PY3_VERSION:
            valid_python_version = False
            if python_version > MAX_SUPPORTED_PY3_VERSION:
                support_message = 'not currently supported by gcloud'
            else:
                support_message = 'no longer supported by gcloud'
            error_message = 'You are running gcloud with Python {python_version}, which is {support_message}.\nInstall a compatible version of Python {min_python_version}-{max_python_version} and set the CLOUDSDK_PYTHON environment variable to point to it.'.format(python_version=python_version_string(python_version), support_message=support_message, min_python_version=python_version_string(MIN_SUPPORTED_PY3_VERSION), max_python_version=python_version_string(MAX_SUPPORTED_PY3_VERSION))
        else:
            valid_python_version = True
            error_message = 'This usually indicates corruption in your gcloud installation or problems with your Python interpreter.\n\nPlease verify that the following is the path to a working Python {min_python_version}-{max_python_version} executable:\n    {executable}\n\nIf it is not, please set the CLOUDSDK_PYTHON environment variable to point to a working Python executable.'.format(executable=sys.executable, min_python_version=python_version_string(MIN_SUPPORTED_PY3_VERSION), max_python_version=python_version_string(MAX_SUPPORTED_PY3_VERSION))
        sys.stderr.write('ERROR: gcloud failed to load. {error_message}\n\nIf you are still experiencing problems, please reinstall the Google Cloud CLI using the instructions here:\n    https://cloud.google.com/sdk/docs/install\n'.format(error_message=error_message))
        if valid_python_version:
            import traceback
            sys.stderr.write('\n\n{}\n'.format('\n'.join(traceback.format_exc().splitlines())))
        sys.exit(1)