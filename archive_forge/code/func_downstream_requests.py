import os
import shutil
import subprocess
import nox
@nox.session(python=['2.7', '3.9'])
def downstream_requests(session):
    root = os.getcwd()
    tmp_dir = session.create_tmp()
    session.cd(tmp_dir)
    git_clone(session, 'https://github.com/psf/requests')
    session.chdir('requests')
    session.run('git', 'apply', f'{root}/ci/requests.patch', external=True)
    session.run('git', 'rev-parse', 'HEAD', external=True)
    session.install('.[socks]', silent=False)
    session.install('-r', 'requirements-dev.txt', silent=False)
    session.cd(root)
    session.install('.', silent=False)
    session.cd(f'{tmp_dir}/requests')
    session.run('pytest', 'tests')