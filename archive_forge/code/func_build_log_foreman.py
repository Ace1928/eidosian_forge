from __future__ import (absolute_import, division, print_function)
import os
from datetime import datetime
from collections import defaultdict
import json
import time
from ansible.module_utils._text import to_text
from ansible.module_utils.parsing.convert_bool import boolean as to_bool
from ansible.plugins.callback import CallbackBase
def build_log_foreman(data_list):
    """
    Transform the internal log structure to one accepted by Foreman's
    config_report API.
    """
    for data in data_list:
        result = data.pop('result')
        task = data.pop('task')
        result['failed'] = data.get('failed')
        result['module'] = task.get('action')
        if data.get('failed'):
            level = 'err'
        elif result.get('changed'):
            level = 'notice'
        else:
            level = 'info'
        yield {'log': {'sources': {'source': task.get('name')}, 'messages': {'message': json.dumps(result, sort_keys=True)}, 'level': level}}