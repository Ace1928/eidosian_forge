from __future__ import absolute_import, division, print_function
import os
import subprocess
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.database import mysql_quote_identifier
from ansible_collections.community.mysql.plugins.module_utils.mysql import mysql_connect, mysql_driver, mysql_driver_fail_msg, mysql_common_argument_spec
from ansible.module_utils.six.moves import shlex_quote
from ansible.module_utils._text import to_native
def db_dump(module, host, user, password, db_name, target, all_databases, port, config_file, socket=None, ssl_cert=None, ssl_key=None, ssl_ca=None, single_transaction=None, quick=None, ignore_tables=None, hex_blob=None, encoding=None, force=False, master_data=0, skip_lock_tables=False, dump_extra_args=None, unsafe_password=False, restrict_config_file=False, check_implicit_admin=False, pipefail=False):
    cmd = module.get_bin_path('mysqldump', True)
    if config_file:
        if restrict_config_file:
            cmd += ' --defaults-file=%s' % shlex_quote(config_file)
        else:
            cmd += ' --defaults-extra-file=%s' % shlex_quote(config_file)
    if check_implicit_admin:
        cmd += " --user=root --password=''"
    else:
        if user is not None:
            cmd += ' --user=%s' % shlex_quote(user)
        if password is not None:
            if not unsafe_password:
                cmd += ' --password=%s' % shlex_quote(password)
            else:
                cmd += ' --password=%s' % password
    if ssl_cert is not None:
        cmd += ' --ssl-cert=%s' % shlex_quote(ssl_cert)
    if ssl_key is not None:
        cmd += ' --ssl-key=%s' % shlex_quote(ssl_key)
    if ssl_ca is not None:
        cmd += ' --ssl-ca=%s' % shlex_quote(ssl_ca)
    if force:
        cmd += ' --force'
    if socket is not None:
        cmd += ' --socket=%s' % shlex_quote(socket)
    else:
        cmd += ' --host=%s --port=%i' % (shlex_quote(host), port)
    if all_databases:
        cmd += ' --all-databases'
    elif len(db_name) > 1:
        cmd += ' --databases {0}'.format(' '.join(db_name))
    else:
        cmd += ' %s' % shlex_quote(' '.join(db_name))
    if skip_lock_tables:
        cmd += ' --skip-lock-tables'
    if encoding is not None and encoding != '':
        cmd += ' --default-character-set=%s' % shlex_quote(encoding)
    if single_transaction:
        cmd += ' --single-transaction=true'
    if quick:
        cmd += ' --quick'
    if ignore_tables:
        for an_ignored_table in ignore_tables:
            cmd += ' --ignore-table={0}'.format(an_ignored_table)
    if hex_blob:
        cmd += ' --hex-blob'
    if master_data:
        cmd += ' --master-data=%s' % master_data
    if dump_extra_args is not None:
        cmd += ' ' + dump_extra_args
    path = None
    if os.path.splitext(target)[-1] == '.gz':
        path = module.get_bin_path('gzip', True)
    elif os.path.splitext(target)[-1] == '.bz2':
        path = module.get_bin_path('bzip2', True)
    elif os.path.splitext(target)[-1] == '.xz':
        path = module.get_bin_path('xz', True)
    if path:
        cmd = '%s | %s > %s' % (cmd, path, shlex_quote(target))
        if pipefail:
            cmd = 'set -o pipefail && ' + cmd
    else:
        cmd += ' > %s' % shlex_quote(target)
    executed_commands.append(cmd)
    if pipefail:
        rc, stdout, stderr = module.run_command(cmd, use_unsafe_shell=True, executable='bash')
    else:
        rc, stdout, stderr = module.run_command(cmd, use_unsafe_shell=True)
    return (rc, stdout, stderr)