from __future__ import absolute_import, division, print_function
import hmac
import itertools
import re
import traceback
from base64 import b64decode
from hashlib import md5, sha256
from ansible.module_utils._text import to_bytes, to_native, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils import \
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
from ansible_collections.community.postgresql.plugins.module_utils.version import \
def parse_role_attrs(role_attr_flags, srv_version):
    """
    Parse role attributes string for user creation.
    Format:

        attributes[,attributes,...]

    Where:

        attributes := CREATEDB,CREATEROLE,NOSUPERUSER,...
        [ "[NO]SUPERUSER","[NO]CREATEROLE", "[NO]CREATEDB",
                            "[NO]INHERIT", "[NO]LOGIN", "[NO]REPLICATION",
                            "[NO]BYPASSRLS" ]

    Note: "[NO]BYPASSRLS" role attribute introduced in 9.5
    Note: "[NO]CREATEUSER" role attribute is deprecated.

    """
    flags = frozenset((role.upper() for role in role_attr_flags.split(',') if role))
    valid_flags = frozenset(itertools.chain(FLAGS, get_valid_flags_by_version(srv_version)))
    valid_flags = frozenset(itertools.chain(valid_flags, ('NO%s' % flag for flag in valid_flags)))
    if not flags.issubset(valid_flags):
        raise InvalidFlagsError('Invalid role_attr_flags specified: %s' % ' '.join(flags.difference(valid_flags)))
    return ' '.join(flags)