import ipaddress
import socket
from openstack import _log
from openstack import exceptions
from openstack import utils
def get_hostvars_from_server(cloud, server, mounts=None):
    """Expand additional server information useful for ansible inventory.

    Variables in this function may make additional cloud queries to flesh out
    possibly interesting info, making it more expensive to call than
    expand_server_vars if caching is not set up. If caching is set up,
    the extra cost should be minimal.
    """
    server_vars = obj_to_munch(add_server_interfaces(cloud, server))
    flavor_id = server['flavor'].get('id')
    if flavor_id:
        flavor_name = cloud.get_flavor_name(flavor_id)
        if flavor_name:
            server_vars['flavor']['name'] = flavor_name
    elif 'original_name' in server['flavor']:
        server_vars['flavor']['name'] = server['flavor']['original_name']
    expand_server_security_groups(cloud, server)
    if str(server['image']) == server['image']:
        image_id = server['image']
        server_vars['image'] = dict(id=image_id)
    else:
        image_id = server['image'].get('id', None)
    if image_id:
        image_name = cloud.get_image_name(image_id)
        if image_name:
            server_vars['image']['name'] = image_name
    if hasattr(server_vars['image'], 'to_dict'):
        server_vars['image'] = server_vars['image'].to_dict(computed=False)
    volumes = []
    if cloud.has_service('volume'):
        try:
            for volume in cloud.get_volumes(server):
                volume['device'] = volume['attachments'][0]['device']
                volumes.append(volume)
        except exceptions.SDKException:
            pass
    server_vars['volumes'] = volumes
    if mounts:
        for mount in mounts:
            for vol in server_vars['volumes']:
                if vol['display_name'] == mount['display_name']:
                    if 'mount' in mount:
                        vol['mount'] = mount['mount']
    return server_vars