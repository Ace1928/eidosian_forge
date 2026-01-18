from __future__ import (absolute_import, division, print_function)
import operator
from ansible.module_utils.compat.version import LooseVersion
from ansible.utils.version import SemanticVersion
def meets_requirements(version, requirements):
    """Verify if a given version satisfies all the requirements.

    Supported version identifiers are:
      * '=='
      * '!='
      * '>'
      * '>='
      * '<'
      * '<='
      * '*'

    Each requirement is delimited by ','.
    """
    op_map = {'!=': operator.ne, '==': operator.eq, '=': operator.eq, '>=': operator.ge, '>': operator.gt, '<=': operator.le, '<': operator.lt}
    for req in requirements.split(','):
        op_pos = 2 if len(req) > 1 and req[1] == '=' else 1
        op = op_map.get(req[:op_pos])
        requirement = req[op_pos:]
        if not op:
            requirement = req
            op = operator.eq
        if requirement == '*' or version == '*':
            continue
        if not op(SemanticVersion(version), SemanticVersion.from_loose_version(LooseVersion(requirement))):
            break
    else:
        return True
    return False