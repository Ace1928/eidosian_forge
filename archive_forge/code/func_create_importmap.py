import logging
from os_ken.services.protocols.bgp.api.base import register
from os_ken.services.protocols.bgp.core_manager import CORE_MANAGER
from os_ken.services.protocols.bgp.core_managers.import_map_manager\
from os_ken.services.protocols.bgp.rtconf.base import RuntimeConfigError
@register(name='importmap.create')
def create_importmap(type, action, name, value, route_family=None):
    if action != 'drop':
        raise RuntimeConfigError('Unknown action. For now we only support "drop" action.')
    if type not in ('prefix_match', 'rt_match'):
        raise RuntimeConfigError('Unknown type. We support only "prefix_match" and "rt_match".')
    if type == 'prefix_match':
        return _create_prefix_match_importmap(name, value, route_family)
    elif type == 'rt_match':
        return _create_rt_match_importmap(name, value)