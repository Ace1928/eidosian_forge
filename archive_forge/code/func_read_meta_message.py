import string
import struct
import time
from numbers import Integral
from ..messages import SPEC_BY_STATUS, Message
from .meta import MetaMessage, build_meta_message, encode_variable_int, meta_charset
from .tracks import MidiTrack, fix_end_of_track, merge_tracks
from .units import tick2second
def read_meta_message(infile, delta):
    meta_type = read_byte(infile)
    length = read_variable_int(infile)
    data = read_bytes(infile, length)
    return build_meta_message(meta_type, data, delta)