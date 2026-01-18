import logging
from oslo_vmware import vim_util
def get_port_group_spec(session, name, vlan_id, trunk_mode=False):
    """Gets the port group spec for a distributed port group

    :param session: vCenter soap session
    :param name: the name of the port group
    :param vlan_id: vlan_id for the port
    :param trunk_mode: indicates if the port will have trunk mode or use
                       specific tag above
    :returns: The configuration for a port group.
    """
    client_factory = session.vim.client.factory
    pg_spec = client_factory.create('ns0:DVPortgroupConfigSpec')
    pg_spec.name = name
    pg_spec.type = 'ephemeral'
    config = client_factory.create('ns0:VMwareDVSPortSetting')
    if trunk_mode:
        config.vlan = get_trunk_vlan_spec(session)
    elif vlan_id:
        config.vlan = get_vlan_spec(session, vlan_id)
    pg_spec.defaultPortConfig = config
    return pg_spec