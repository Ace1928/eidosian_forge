import os
import pathlib
import subprocess
import shutil
import tempfile
from nox.command import which
import nox
def copy_credentials(credentials_path):
    """Copies credentials into the SDK root as the application default
    credentials."""
    dest = CLOUD_SDK_ROOT.joinpath('application_default_credentials.json')
    if dest.exists():
        dest.unlink()
    shutil.copyfile(pathlib.Path(credentials_path), dest)