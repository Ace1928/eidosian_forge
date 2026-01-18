from typing import Union
from warnings import warn
from .low_level import *
def new_header(msg_type):
    return Header(Endianness.little, msg_type, flags=0, protocol_version=1, body_length=-1, serial=-1, fields={})