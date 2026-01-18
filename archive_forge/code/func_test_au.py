import warnings
from collections import namedtuple
def test_au(h, f):
    """AU and SND files"""
    if h.startswith(b'.snd'):
        func = get_long_be
    elif h[:4] in (b'\x00ds.', b'dns.'):
        func = get_long_le
    else:
        return None
    filetype = 'au'
    hdr_size = func(h[4:8])
    data_size = func(h[8:12])
    encoding = func(h[12:16])
    rate = func(h[16:20])
    nchannels = func(h[20:24])
    sample_size = 1
    if encoding == 1:
        sample_bits = 'U'
    elif encoding == 2:
        sample_bits = 8
    elif encoding == 3:
        sample_bits = 16
        sample_size = 2
    else:
        sample_bits = '?'
    frame_size = sample_size * nchannels
    if frame_size:
        nframe = data_size / frame_size
    else:
        nframe = -1
    return (filetype, rate, nchannels, nframe, sample_bits)