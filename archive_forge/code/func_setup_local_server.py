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
def setup_local_server(task_name, task_files_to_copy=None, tmp_dir=parent_dir):
    global server_process
    server_dir = compile_server(task_files_to_copy, task_name, tmp_dir)
    print('Local: Starting server...')
    os.chdir(os.path.join(server_dir, 'server'))
    packages_installed = subprocess.call(['npm', 'install'])
    if packages_installed != 0:
        raise Exception('please make sure npm is installed, otherwise view the above error for more info.')
    server_process = subprocess.Popen(['node', 'server.js'])
    time.sleep(1)
    print('Server running locally with pid {}.'.format(server_process.pid))
    host = input('Please enter the public server address, like https://hostname.com: ')
    port = input('Please enter the port given above, likely 3000: ')
    return '{}:{}'.format(host, port)