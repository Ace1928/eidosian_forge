import logging
import os
import re
import subprocess
import sys
from mlflow.exceptions import MlflowException
from mlflow.models import FlavorBackend
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.string_utils import quote
def can_score_model(self):
    process = subprocess.Popen(['Rscript', '--version'], close_fds=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, _ = process.communicate()
    if process.wait() != 0:
        return False
    version = self.version_pattern.search(stdout.decode('utf-8'))
    if not version:
        return False
    version = [int(x) for x in version.group(1).split('.')]
    return version[0] > 3 or (version[0] == 3 and version[1] >= 3)