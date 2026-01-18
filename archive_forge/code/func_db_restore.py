from __future__ import absolute_import, division, print_function
import os
import subprocess
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import shlex_quote
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def db_restore(module, target, target_opts='', db=None, user=None, password=None, host=None, port=None, **kw):
    flags = login_flags(db, host, port, user)
    comp_prog_path = None
    cmd = module.get_bin_path('psql', True)
    if os.path.splitext(target)[-1] == '.sql':
        flags.append(' --file={0}'.format(target))
    elif os.path.splitext(target)[-1] == '.tar':
        flags.append(' --format=Tar')
        cmd = module.get_bin_path('pg_restore', True)
    elif os.path.splitext(target)[-1] == '.pgc':
        flags.append(' --format=Custom')
        cmd = module.get_bin_path('pg_restore', True)
    elif os.path.splitext(target)[-1] == '.dir':
        flags.append(' --format=Directory')
        cmd = module.get_bin_path('pg_restore', True)
    elif os.path.splitext(target)[-1] == '.gz':
        comp_prog_path = module.get_bin_path('zcat', True)
    elif os.path.splitext(target)[-1] == '.bz2':
        comp_prog_path = module.get_bin_path('bzcat', True)
    elif os.path.splitext(target)[-1] == '.xz':
        comp_prog_path = module.get_bin_path('xzcat', True)
    cmd += ''.join(flags)
    if target_opts:
        cmd += ' {0} '.format(target_opts)
    if comp_prog_path:
        env = os.environ.copy()
        if password:
            env = {'PGPASSWORD': password}
        p1 = subprocess.Popen([comp_prog_path, target], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p2 = subprocess.Popen(cmd, stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=env)
        stdout2, stderr2 = p2.communicate()
        p1.stdout.close()
        p1.wait()
        if p1.returncode != 0:
            stderr1 = p1.stderr.read()
            return (p1.returncode, '', stderr1, 'cmd: ****')
        else:
            return (p2.returncode, '', stderr2, 'cmd: ****')
    elif '--format=Directory' in cmd:
        cmd = '{0} {1}'.format(cmd, shlex_quote(target))
    else:
        cmd = '{0} < {1}'.format(cmd, shlex_quote(target))
    return do_with_password(module, cmd, password)