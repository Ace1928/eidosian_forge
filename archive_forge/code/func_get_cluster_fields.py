from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def get_cluster_fields(cluster):
    """ Find fields for cluster """
    fields = cluster.get_fields(from_cache=True, raw_value=True)
    created_at, created_at_timezone = unixMillisecondsToDate(fields.get('created_at', None))
    field_dict = dict(hosts=[], id=cluster.id, created_at=created_at, created_at_timezone=created_at_timezone)
    hosts = cluster.get_hosts()
    for host in hosts:
        host_dict = {'host_id': host.id, 'host_name': host.get_name()}
        field_dict['hosts'].append(host_dict)
    return field_dict