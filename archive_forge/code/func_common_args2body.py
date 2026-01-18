import argparse
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
def common_args2body(parsed_args):
    body = {}
    neutronv20.update_dict(parsed_args, body, ['name', 'description', 'shared', 'tenant_id', 'source_ip_address', 'destination_ip_address', 'source_port', 'destination_port', 'action', 'enabled', 'ip_version'])
    protocol = parsed_args.protocol
    if protocol:
        if protocol == 'any':
            protocol = None
        body['protocol'] = protocol
    return body