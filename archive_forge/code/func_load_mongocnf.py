from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib  # pylint: disable=unused-import:
from ansible.module_utils.six.moves import configparser
from ansible.module_utils._text import to_native
import traceback
import os
import ssl as ssl_lib
def load_mongocnf():
    config = configparser.RawConfigParser()
    mongocnf = os.path.expanduser('~/.mongodb.cnf')
    try:
        config.readfp(open(mongocnf))
    except (configparser.NoOptionError, IOError):
        return False
    creds = dict(user=config.get('client', 'user'), password=config.get('client', 'pass'))
    return creds