from __future__ import (absolute_import, division, print_function)
import json
import os
from functools import partial
from ansible.module_utils._text import to_native
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common.validation import check_type_dict, safe_eval
def normalize_extattrs(value):
    """ Normalize extattrs field to expected format
    The module accepts extattrs as key/value pairs.  This method will
    transform the key/value pairs into a structure suitable for
    sending across WAPI in the format of:
        extattrs: {
            key: {
                value: <value>
            }
        }
    """
    return dict([(k, {'value': v}) for k, v in iteritems(value)])