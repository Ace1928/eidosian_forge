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
def setup_heroku_server(task_name, task_files_to_copy=None, heroku_team=None, use_hobby=False, tmp_dir=parent_dir):
    print('Locating heroku...')
    os_name = None
    bit_architecture = None
    platform_info = platform.platform()
    if 'Darwin' in platform_info:
        os_name = 'darwin'
    elif 'Linux' in platform_info:
        os_name = 'linux'
    else:
        os_name = 'windows'
    bit_architecture_info = platform.architecture()[0]
    if '64bit' in bit_architecture_info:
        bit_architecture = 'x64'
    else:
        bit_architecture = 'x86'
    existing_heroku_directory_names = glob.glob(os.path.join(tmp_dir, 'heroku-cli-*'))
    if len(existing_heroku_directory_names) == 0:
        print('Getting heroku')
        if os.path.exists(os.path.join(tmp_dir, 'heroku.tar.gz')):
            os.remove(os.path.join(tmp_dir, 'heroku.tar.gz'))
        os.chdir(tmp_dir)
        sh.wget(shlex.split('{}-{}-{}.tar.gz -O heroku.tar.gz'.format(heroku_url, os_name, bit_architecture)))
        sh.tar(shlex.split('-xvzf heroku.tar.gz'))
    heroku_directory_name = glob.glob(os.path.join(tmp_dir, 'heroku-cli-*'))[0]
    heroku_directory_path = os.path.join(tmp_dir, heroku_directory_name)
    heroku_executable_path = os.path.join(heroku_directory_path, 'bin', 'heroku')
    server_dir = compile_server(task_files_to_copy, task_name, tmp_dir)
    print('Heroku: Starting server...')
    heroku_server_directory_path = os.path.join(server_dir, 'server')
    os.chdir(heroku_server_directory_path)
    sh.git('init')
    heroku_user_identifier = None
    while not heroku_user_identifier:
        try:
            subprocess.check_output(shlex.split(heroku_executable_path + ' auth:token'))
            heroku_user_identifier = netrc.netrc(os.path.join(os.path.expanduser('~'), '.netrc')).hosts['api.heroku.com'][0]
        except subprocess.CalledProcessError:
            raise SystemExit('A free Heroku account is required for launching MTurk tasks. Please register at https://signup.heroku.com/ and run `{} login` at the terminal to login to Heroku, and then run this program again.'.format(heroku_executable_path))
    heroku_app_name = '{}-{}-{}'.format(user_name, task_name, hashlib.md5(heroku_user_identifier.encode('utf-8')).hexdigest())[:30]
    while heroku_app_name[-1] == '-':
        heroku_app_name = heroku_app_name[:-1]
    try:
        if heroku_team is not None:
            subprocess.check_output(shlex.split('{} create {} --team {}'.format(heroku_executable_path, heroku_app_name, heroku_team)))
        else:
            subprocess.check_output(shlex.split('{} create {}'.format(heroku_executable_path, heroku_app_name)))
    except subprocess.CalledProcessError:
        sh.rm(shlex.split('-rf {}'.format(heroku_server_directory_path)))
        raise SystemExit('You have hit your limit on concurrent apps with heroku, which are required to run multiple concurrent tasks.\nPlease wait for some of your existing tasks to complete. If you have no tasks running, login to heroku and delete some of the running apps or verify your account to allow more concurrent apps')
    try:
        subprocess.check_output(shlex.split('{} features:enable http-session-affinity'.format(heroku_executable_path)))
    except subprocess.CalledProcessError:
        pass
    os.chdir(heroku_server_directory_path)
    sh.git(shlex.split('add -A'))
    sh.git(shlex.split('commit -m "app"'))
    sh.git(shlex.split('push -f heroku master'))
    subprocess.check_output(shlex.split('{} ps:scale web=1'.format(heroku_executable_path)))
    if use_hobby:
        try:
            subprocess.check_output(shlex.split('{} dyno:type Hobby'.format(heroku_executable_path)))
        except subprocess.CalledProcessError:
            delete_heroku_server(task_name)
            sh.rm(shlex.split('-rf {}'.format(heroku_server_directory_path)))
            raise SystemExit('Server launched with hobby flag but account cannot create hobby servers.')
    os.chdir(parent_dir)
    if os.path.exists(os.path.join(tmp_dir, 'heroku.tar.gz')):
        os.remove(os.path.join(tmp_dir, 'heroku.tar.gz'))
    sh.rm(shlex.split('-rf {}'.format(server_dir)))
    return 'https://{}.herokuapp.com'.format(heroku_app_name)