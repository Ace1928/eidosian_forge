from ovs.flow.decoders import (
from ovs.flow.kv import (
from ovs.flow.list import nested_list_decoder, ListDecoders
from ovs.flow.ofp_fields import field_decoders, field_aliases
def learn_field_decoding_free(key):
    """Decodes the free fields found in the learn action.
        Free fields indicate that the filed is to be copied from the original.
        In order to express that in a dictionary, return the fieldspec as
        value. So, the free fild NXM_OF_IP_SRC[], is encoded as:
            "NXM_OF_IP_SRC[]": {
                "field": "NXM_OF_IP_SRC"
            }
        That way we also ensure the actual free key is correct.
        """
    key_field = decode_field(key)
    return (key, key_field)