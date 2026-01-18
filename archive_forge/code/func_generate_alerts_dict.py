from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from datetime import datetime
import time
def generate_alerts_dict(array):
    alerts_info = {}
    alerts = list(array.get_alerts().items)
    for alert in range(0, len(alerts)):
        name = alerts[alert].name
        try:
            notified_time = alerts[alert].notified / 1000
            notified_datetime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(notified_time))
        except AttributeError:
            notified_datetime = ''
        try:
            closed_time = alerts[alert].closed / 1000
            closed_datetime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(closed_time))
        except AttributeError:
            closed_datetime = ''
        try:
            updated_time = alerts[alert].updated / 1000
            updated_datetime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(updated_time))
        except AttributeError:
            updated_datetime = ''
        try:
            created_time = alerts[alert].created / 1000
            created_datetime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(created_time))
        except AttributeError:
            updated_datetime = ''
        alerts_info[name] = {'flagged': alerts[alert].flagged, 'category': alerts[alert].category, 'code': alerts[alert].code, 'issue': alerts[alert].issue, 'kb_url': alerts[alert].knowledge_base_url, 'summary': alerts[alert].summary, 'id': alerts[alert].id, 'state': alerts[alert].state, 'severity': alerts[alert].severity, 'component_name': alerts[alert].component_name, 'component_type': alerts[alert].component_type, 'created': created_datetime, 'closed': closed_datetime, 'notified': notified_datetime, 'updated': updated_datetime, 'actual': getattr(alerts[alert], 'actual', ''), 'expected': getattr(alerts[alert], 'expected', '')}
    return alerts_info