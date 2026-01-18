from __future__ import (absolute_import, division, print_function)
import time
def get_monitoring_policy(oneandone_conn, monitoring_policy, full_object=False):
    """
    Validates the monitoring policy exists by ID or name.
    Return the monitoring policy ID.
    """
    for _monitoring_policy in oneandone_conn.list_monitoring_policies():
        if monitoring_policy in (_monitoring_policy['name'], _monitoring_policy['id']):
            if full_object:
                return _monitoring_policy
            return _monitoring_policy['id']