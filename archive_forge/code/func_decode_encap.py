from ovs.flow.decoders import (
from ovs.flow.kv import (
from ovs.flow.list import nested_list_decoder, ListDecoders
from ovs.flow.ofp_fields import field_decoders, field_aliases
def decode_encap(value):
    """Decodes encap action. Examples:
    encap(ethernet)
    encap(nsh(md_type=2,tlv(0x1000,10,0x12345678)))

    The generated dict has the following keys: "header", "props", e.g:
        {
            "header": "ethernet",
        }
        {
            "header": "nsh",
            "props": {
                "md_type": 2,
                "tlv": {
                    "class": 0x100,
                    "type": 10,
                    "value": 0x123456
                }
            }
        }
    """

    def free_hdr_decoder(free_val):
        if free_val not in ['ethernet', 'mpls', 'mpls_mc', 'nsh']:
            raise ValueError('Malformed encap action. Unkown header: {}'.format(free_val))
        return ('header', free_val)
    parser = KVParser(value, KVDecoders({'nsh': nested_kv_decoder(KVDecoders({'md_type': decode_default, 'tlv': nested_list_decoder(ListDecoders([('class', decode_int), ('type', decode_int), ('value', decode_int)]))}))}, default_free=free_hdr_decoder))
    parser.parse()
    if len(parser.kv()) > 1:
        raise ValueError('Malformed encap action: {}'.format(value))
    result = {}
    if parser.kv()[0].key == 'header':
        result['header'] = parser.kv()[0].value
    elif parser.kv()[0].key == 'nsh':
        result['header'] = 'nsh'
        result['props'] = parser.kv()[0].value
    return result