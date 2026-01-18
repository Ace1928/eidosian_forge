from __future__ import absolute_import, division, print_function
import os
import subprocess
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import shlex_quote
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def login_flags(db, host, port, user, db_prefix=True):
    """
    returns a list of connection argument strings each prefixed
    with a space and quoted where necessary to later be combined
    in a single shell string with `"".join(rv)`

    db_prefix determines if "--dbname" is prefixed to the db argument,
    since the argument was introduced in 9.3.
    """
    flags = []
    if db:
        if db_prefix:
            flags.append(' --dbname={0}'.format(shlex_quote(db)))
        else:
            flags.append(' {0}'.format(shlex_quote(db)))
    if host:
        flags.append(' --host={0}'.format(host))
    if port:
        flags.append(' --port={0}'.format(port))
    if user:
        flags.append(' --username={0}'.format(user))
    return flags