import json
import netaddr
import re
def decode_nat(value):
    """Decodes the 'nat' keyword of the ct action.

    The format is:
        nat
            Flag format.
        nat(type=addrs[:ports][,flag]...)
            Full format where the address-port range has the same format as
            the one described in decode_ip_port_range.

    Examples:
        nat(src=0.0.0.0)
        nat(src=0.0.0.0,persistent)
        nat(dst=192.168.1.0-192.168.1.253:4000-5000)
        nat(dst=192.168.1.0-192.168.1.253,hash)
        nat(dst=[fe80::f150]-[fe80::f15f]:255-300)
    """
    if not value:
        return True
    result = dict()
    type_parts = value.split('=')
    result['type'] = type_parts[0]
    if len(type_parts) > 1:
        value_parts = type_parts[1].split(',')
        if len(type_parts) != 2:
            raise ValueError('Malformed nat action: %s' % value)
        ip_port_range = decode_ip_port_range(value_parts[0])
        result = {'type': type_parts[0], **ip_port_range}
        for flag in value_parts[1:]:
            result[flag] = True
    return result