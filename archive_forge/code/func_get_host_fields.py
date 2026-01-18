from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
def get_host_fields(host):
    """ Get host fields """
    fields = host.get_fields(from_cache=True, raw_value=True)
    created_at, created_at_timezone = unixMillisecondsToDate(fields.get('created_at', None))
    field_dict = dict(created_at=created_at, created_at_timezone=created_at_timezone, id=host.id, iqns=[], luns=[], ports=[], wwns=[])
    luns = host.get_luns()
    for lun in luns:
        field_dict['luns'].append({'lun_id': lun.id, 'lun_volume_id': lun.volume.id, 'lun_volume_name': lun.volume.get_name()})
    ports = host.get_ports()
    for port in ports:
        if str(type(port)) == "<class 'infi.dtypes.wwn.WWN'>":
            field_dict['wwns'].append(str(port))
        if str(type(port)) == "<class 'infi.dtypes.iqn.IQN'>":
            field_dict['iqns'].append(str(port))
    return field_dict