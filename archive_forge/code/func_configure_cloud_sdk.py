import os
import pathlib
import subprocess
import shutil
import tempfile
from nox.command import which
import nox
def configure_cloud_sdk(session, application_default_credentials, project=False):
    """Installs and configures the Cloud SDK with the given application default
    credentials.

    If project is True, then a project will be set in the active config.
    If it is false, this will ensure no project is set.
    """
    install_cloud_sdk(session)
    session.run(GCLOUD, 'auth', 'activate-service-account', '--key-file', SERVICE_ACCOUNT_FILE)
    if project:
        session.run(GCLOUD, 'config', 'set', 'project', 'example-project')
    else:
        session.run(GCLOUD, 'config', 'unset', 'project')
    session.run(copy_credentials, application_default_credentials)
    session.run(GCLOUD, 'auth', 'application-default', 'print-access-token', silent=True)