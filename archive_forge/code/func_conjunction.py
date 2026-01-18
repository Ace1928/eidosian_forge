import re
import os_ken.exception
from os_ken.lib.ofctl_utils import str_to_int
from os_ken.ofproto import nicira_ext
@classmethod
def conjunction(cls, ofproto, action_str):
    try:
        assert action_str.startswith('conjunction(')
        assert action_str[-1] == ')'
        args = action_str[len('conjunction('):-1].split(',')
        assert len(args) == 2
        id_ = str_to_int(args[0])
        clauses = list(map(str_to_int, args[1].split('/')))
        assert len(clauses) == 2
        return dict(NXActionConjunction={'clause': clauses[0] - 1, 'n_clauses': clauses[1], 'id': id_})
    except Exception:
        raise os_ken.exception.OFPInvalidActionString(action_str=action_str)