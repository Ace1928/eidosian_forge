from __future__ import absolute_import, division, print_function
import collections
import json
from contextlib import contextmanager
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import Connection, ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import (
def load_configuration(module, candidate=None, action='merge', rollback=None, format='xml'):
    if all((candidate is None, rollback is None)):
        module.fail_json(msg='one of candidate or rollback must be specified')
    elif all((candidate is not None, rollback is not None)):
        module.fail_json(msg='candidate and rollback are mutually exclusive')
    if format not in FORMATS:
        module.fail_json(msg='invalid format specified')
    if format == 'json' and action not in JSON_ACTIONS:
        module.fail_json(msg='invalid action for format json')
    elif format in ('text', 'xml') and action not in ACTIONS:
        module.fail_json(msg='invalid action format %s' % format)
    if action == 'set' and format != 'text':
        module.fail_json(msg='format must be text when action is set')
    conn = get_connection(module)
    try:
        if rollback is not None:
            _validate_rollback_id(module, rollback)
            obj = Element('load-configuration', {'rollback': str(rollback)})
            conn.execute_rpc(tostring(obj))
        else:
            return conn.load_configuration(config=candidate, action=action, format=format)
    except ConnectionError as exc:
        module.fail_json(msg=to_text(exc, errors='surrogate_then_replace'))