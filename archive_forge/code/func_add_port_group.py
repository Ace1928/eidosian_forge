import logging
from oslo_vmware import vim_util
def add_port_group(session, dvs_moref, name, vlan_id=None, trunk_mode=False):
    """Add a new port group to the dvs_moref

    :param session: vCenter soap session
    :param dvs_moref: managed DVS object reference
    :param name: the name of the port group
    :param vlan_id: vlan_id for the port
    :param trunk_mode: indicates if the port will have trunk mode or use
                       specific tag above
    :returns: The new portgroup moref
    """
    pg_spec = get_port_group_spec(session, name, vlan_id, trunk_mode=trunk_mode)
    task = session.invoke_api(session.vim, 'CreateDVPortgroup_Task', dvs_moref, spec=pg_spec)
    task_info = session.wait_for_task(task)
    LOG.info('%(name)s create on %(dvs)s with %(value)s.', {'name': name, 'dvs': vim_util.get_moref_value(dvs_moref), 'value': task_info.result.value})
    return task_info.result