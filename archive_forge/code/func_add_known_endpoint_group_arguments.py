from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronv20
def add_known_endpoint_group_arguments(parser, is_create=True):
    parser.add_argument('--name', help=_('Set a name for the endpoint group.'))
    parser.add_argument('--description', help=_('Set a description for the endpoint group.'))
    if is_create:
        parser.add_argument('--type', required=is_create, help=_('Type of endpoints in group (e.g. subnet, cidr, vlan).'))
        parser.add_argument('--value', action='append', dest='endpoints', required=is_create, help=_('Endpoint(s) for the group. Must all be of the same type.'))