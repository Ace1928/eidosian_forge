import string
import struct
import time
from numbers import Integral
from ..messages import SPEC_BY_STATUS, Message
from .meta import MetaMessage, build_meta_message, encode_variable_int, meta_charset
from .tracks import MidiTrack, fix_end_of_track, merge_tracks
from .units import tick2second
def read_sysex(infile, delta, clip=False):
    length = read_variable_int(infile)
    data = read_bytes(infile, length)
    if data and data[0] == 240:
        data = data[1:]
    if data and data[-1] == 247:
        data = data[:-1]
    if clip:
        data = [byte if byte < 127 else 127 for byte in data]
    return Message('sysex', data=data, time=delta)