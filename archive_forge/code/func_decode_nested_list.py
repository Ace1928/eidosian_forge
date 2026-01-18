import re
from ovs.flow.kv import KeyValue, KeyMetadata, ParseError
from ovs.flow.decoders import decode_default
def decode_nested_list(decoders, value, delims=[',']):
    """Decodes a string value that contains a list of elements and returns
    them in a dictionary.

    Args:
        decoders (ListDecoders): The ListDecoders to use.
        value (str): The value string to decode.
        delims (list(str)): Optional, the list of delimiters to use.
    """
    parser = ListParser(value, decoders, delims)
    parser.parse()
    return {kv.key: kv.value for kv in parser.kv()}