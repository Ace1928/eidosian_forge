from __future__ import absolute_import, division, print_function
import os
import re
import traceback
import shutil
import tempfile
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def fromline(self, line):
    """
        split into 'type', 'db', 'usr', 'src', 'mask', 'method', 'options' cols
        """
    if WHITESPACES_RE.sub('', line) == '':
        return
    cols = WHITESPACES_RE.split(line)
    if len(cols) < 4:
        msg = 'Rule {0} has too few columns.'
        raise PgHbaValueError(msg.format(line))
    if cols[0] not in PG_HBA_TYPES:
        msg = 'Rule {0} has unknown type: {1}.'
        raise PgHbaValueError(msg.format(line, cols[0]))
    if cols[0] == 'local':
        cols.insert(3, None)
        cols.insert(3, None)
    if len(cols) < 6:
        cols.insert(4, None)
    elif cols[5] not in PG_HBA_METHODS:
        cols.insert(4, None)
    if cols[5] not in PG_HBA_METHODS:
        raise PgHbaValueError("Rule {0} of '{1}' type has invalid auth-method '{2}'".format(line, cols[0], cols[5]))
    if len(cols) < 7:
        cols.insert(6, None)
    else:
        cols[6] = ' '.join(cols[6:])
    rule = dict(zip(PG_HBA_HDR, cols[:7]))
    for key, value in rule.items():
        if value:
            self[key] = value