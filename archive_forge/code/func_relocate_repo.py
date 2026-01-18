from __future__ import absolute_import, division, print_function
import filecmp
import os
import re
import shlex
import stat
import sys
import shutil
import tempfile
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.six import b, string_types
def relocate_repo(module, result, repo_dir, old_repo_dir, worktree_dir):
    if os.path.exists(repo_dir):
        module.fail_json(msg='Separate-git-dir path %s already exists.' % repo_dir)
    if worktree_dir:
        dot_git_file_path = os.path.join(worktree_dir, '.git')
        try:
            shutil.move(old_repo_dir, repo_dir)
            with open(dot_git_file_path, 'w') as dot_git_file:
                dot_git_file.write('gitdir: %s' % repo_dir)
            result['git_dir_before'] = old_repo_dir
            result['git_dir_now'] = repo_dir
        except (IOError, OSError) as err:
            if os.path.exists(repo_dir):
                shutil.move(repo_dir, old_repo_dir)
            module.fail_json(msg=u'Unable to move git dir. %s' % to_text(err))