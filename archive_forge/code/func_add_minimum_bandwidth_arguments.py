from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.qos import rule as qos_rule
def add_minimum_bandwidth_arguments(parser):
    parser.add_argument('--min-kbps', required=True, type=str, help=_('QoS minimum bandwidth assurance, expressed in kilobits per second.'))
    parser.add_argument('--direction', required=True, type=utils.convert_to_lowercase, choices=['egress'], help=_('Traffic direction.'))