from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
def check_member_attrs(attrs):
    if 'protocol_port' in attrs:
        _validate_TCP_UDP_SCTP_port_range(attrs['protocol_port'], 'protocol-port')
    if 'member_port' in attrs:
        _validate_TCP_UDP_SCTP_port_range(attrs['member_port'], 'member-port')
    if 'weight' in attrs:
        if attrs['weight'] < constants.MIN_WEIGHT or attrs['weight'] > constants.MAX_WEIGHT:
            msg = "Invalid input for field/attribute 'weight', Value: '{weight}'. Value must be between {wmin} and {wmax}.".format(weight=attrs['weight'], wmin=constants.MIN_WEIGHT, wmax=constants.MAX_WEIGHT)
            raise exceptions.InvalidValue(msg)