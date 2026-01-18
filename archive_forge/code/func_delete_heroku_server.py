import getpass
import glob
import hashlib
import netrc
import os
import platform
import sh
import shlex
import shutil
import subprocess
import time
import parlai.mturk.core.dev.shared_utils as shared_utils
def delete_heroku_server(task_name, tmp_dir=parent_dir):
    heroku_directory_name = glob.glob(os.path.join(tmp_dir, 'heroku-cli-*'))[0]
    heroku_directory_path = os.path.join(tmp_dir, heroku_directory_name)
    heroku_executable_path = os.path.join(heroku_directory_path, 'bin', 'heroku')
    heroku_user_identifier = netrc.netrc(os.path.join(os.path.expanduser('~'), '.netrc')).hosts['api.heroku.com'][0]
    heroku_app_name = '{}-{}-{}'.format(user_name, task_name, hashlib.md5(heroku_user_identifier.encode('utf-8')).hexdigest())[:30]
    while heroku_app_name[-1] == '-':
        heroku_app_name = heroku_app_name[:-1]
    print('Heroku: Deleting server: {}'.format(heroku_app_name))
    subprocess.check_output(shlex.split('{} destroy {} --confirm {}'.format(heroku_executable_path, heroku_app_name, heroku_app_name)))