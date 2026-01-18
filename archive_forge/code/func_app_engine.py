import os
import pathlib
import subprocess
import shutil
import tempfile
from nox.command import which
import nox
@nox.session(python=['2.7'])
def app_engine(session):
    if SKIP_GAE_TEST_ENV in os.environ:
        session.log('Skipping App Engine tests.')
        return
    session.install(LIBRARY_DIR)
    project_id = subprocess.check_output(['gcloud', 'config', 'list', 'project', '--format', 'value(core.project)']).decode('utf-8').strip()
    if not project_id:
        session.error('The Cloud SDK must be installed and configured to deploy to App Engine.')
    application_url = GAE_APP_URL_TMPL.format(GAE_TEST_APP_SERVICE, project_id)
    session.chdir(os.path.join(HERE, 'system_tests_sync/app_engine_test_app'))
    session.install(*TEST_DEPENDENCIES_SYNC)
    session.run('pip', 'install', '--target', 'lib', '-r', 'requirements.txt', silent=True)
    session.run('gcloud', 'app', 'deploy', '-q', 'app.yaml')
    session.env['TEST_APP_URL'] = application_url
    session.chdir(HERE)
    default(session, 'system_tests_sync/test_app_engine.py')