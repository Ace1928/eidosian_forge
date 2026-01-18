import functools
import json
import jsonschema
import yaml
from ironicclient import exc
def create_portgroups(client, portgroup_list, node_uuid):
    """Create port groups from dictionaries.

    :param client: ironic client instance.
    :param portgroup_list: list of dictionaries to be POSTed to /portgroups
        endpoint, if some of them contain "ports" key, its content is POSTed
        separately to /ports endpoint.
    :param node_uuid: UUID of a node the port groups should be associated with.
    :returns: array of exceptions encountered during creation.
    """
    errors = []
    for portgroup in portgroup_list:
        portgroup_node_uuid = portgroup.get('node_uuid')
        if portgroup_node_uuid and portgroup_node_uuid != node_uuid:
            errors.append(exc.ClientException('Cannot create a port group as part of node %(node_uuid)s because the port group %(portgroup)s has a different node UUID specified.', {'node_uuid': node_uuid, 'portgroup': portgroup}))
            continue
        portgroup['node_uuid'] = node_uuid
        portgroup_uuid, error = create_single_portgroup(client, **portgroup)
        if error:
            errors.append(error)
        ports = portgroup.get('ports')
        if ports is not None and portgroup_uuid is not None:
            errors.extend(create_ports(client, ports, node_uuid, portgroup_uuid=portgroup_uuid))
    return errors