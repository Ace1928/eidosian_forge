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
def delete_local_server(task_name):
    global server_process
    print('Terminating server')
    server_process.terminate()
    server_process.wait()
    print('Cleaning temp directory')
    local_server_directory_path = os.path.join(parent_dir, '{}_{}'.format(local_server_directory_name, task_name))
    sh.rm(shlex.split('-rf {}'.format(local_server_directory_path)))