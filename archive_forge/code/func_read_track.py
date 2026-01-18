import string
import struct
import time
from numbers import Integral
from ..messages import SPEC_BY_STATUS, Message
from .meta import MetaMessage, build_meta_message, encode_variable_int, meta_charset
from .tracks import MidiTrack, fix_end_of_track, merge_tracks
from .units import tick2second
def read_track(infile, debug=False, clip=False):
    track = MidiTrack()
    name, size = read_chunk_header(infile)
    if name != b'MTrk':
        raise OSError('no MTrk header at start of track')
    if debug:
        _dbg(f'-> size={size}')
        _dbg()
    start = infile.tell()
    last_status = None
    while True:
        if infile.tell() - start == size:
            break
        if debug:
            _dbg('Message:')
        delta = read_variable_int(infile)
        if debug:
            _dbg(f'-> delta={delta}')
        status_byte = read_byte(infile)
        if status_byte < 128:
            if last_status is None:
                raise OSError('running status without last_status')
            peek_data = [status_byte]
            status_byte = last_status
        else:
            if status_byte != 255:
                last_status = status_byte
            peek_data = []
        if status_byte == 255:
            msg = read_meta_message(infile, delta)
        elif status_byte in [240, 247]:
            msg = read_sysex(infile, delta, clip)
        else:
            msg = read_message(infile, status_byte, peek_data, delta, clip)
        track.append(msg)
        if debug:
            _dbg(f'-> {msg!r}')
            _dbg()
    return track