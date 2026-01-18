import string
import struct
import time
from numbers import Integral
from ..messages import SPEC_BY_STATUS, Message
from .meta import MetaMessage, build_meta_message, encode_variable_int, meta_charset
from .tracks import MidiTrack, fix_end_of_track, merge_tracks
from .units import tick2second
def read_variable_int(infile):
    delta = 0
    while True:
        byte = read_byte(infile)
        delta = delta << 7 | byte & 127
        if byte < 128:
            return delta