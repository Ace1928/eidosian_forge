from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.qos import policy as qos_policy
def add_policy_argument(parser):
    parser.add_argument('policy', metavar='QOS_POLICY', help=_('ID or name of the QoS policy.'))