from ovs.flow.decoders import (
from ovs.flow.kv import (
from ovs.flow.list import nested_list_decoder, ListDecoders
from ovs.flow.ofp_fields import field_decoders, field_aliases
def decode_controller(value):
    """Decodes the controller action."""
    if not value:
        return KeyValue('output', {'port': 'CONTROLLER'})
    else:
        try:
            max_len = int(value)
            return {'max_len': max_len}
        except ValueError:
            pass
        return nested_kv_decoder(KVDecoders({'max_len': decode_int, 'reason': decode_default, 'id': decode_int, 'userdata': decode_default, 'pause': decode_flag}))(value)