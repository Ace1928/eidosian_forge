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
import parlai.mturk.core.shared_utils as shared_utils
def setup_legacy_server(task_name, task_files_to_copy, local=False, heroku_team=None, use_hobby=False, legacy=True, tmp_dir=parent_dir):
    if local:
        return setup_local_server(task_name, task_files_to_copy=task_files_to_copy)
    return setup_legacy_heroku_server(task_name, task_files_to_copy=task_files_to_copy, heroku_team=heroku_team, use_hobby=use_hobby, tmp_dir=tmp_dir)