import re
from ovs.flow.kv import KeyValue, KeyMetadata, ParseError
from ovs.flow.decoders import decode_default
def nested_list_decoder(decoders=None, delims=[',']):
    """Helper function that creates a nested list decoder with given
    ListDecoders and delimiters.
    """

    def decoder(value):
        return decode_nested_list(decoders, value, delims)
    return decoder