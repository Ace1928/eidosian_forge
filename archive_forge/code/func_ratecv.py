import math
import struct
from ctypes import create_string_buffer
def ratecv(cp, size, nchannels, inrate, outrate, state, weightA=1, weightB=0):
    _check_params(len(cp), size)
    if nchannels < 1:
        raise error('# of channels should be >= 1')
    bytes_per_frame = size * nchannels
    frame_count = len(cp) / bytes_per_frame
    if bytes_per_frame / nchannels != size:
        raise OverflowError('width * nchannels too big for a C int')
    if weightA < 1 or weightB < 0:
        raise error('weightA should be >= 1, weightB should be >= 0')
    if len(cp) % bytes_per_frame != 0:
        raise error('not a whole number of frames')
    if inrate <= 0 or outrate <= 0:
        raise error('sampling rate not > 0')
    d = gcd(inrate, outrate)
    inrate /= d
    outrate /= d
    prev_i = [0] * nchannels
    cur_i = [0] * nchannels
    if state is None:
        d = -outrate
    else:
        d, samps = state
        if len(samps) != nchannels:
            raise error('illegal state argument')
        prev_i, cur_i = zip(*samps)
        prev_i, cur_i = (list(prev_i), list(cur_i))
    q = frame_count / inrate
    ceiling = (q + 1) * outrate
    nbytes = ceiling * bytes_per_frame
    result = create_string_buffer(nbytes)
    samples = _get_samples(cp, size)
    out_i = 0
    while True:
        while d < 0:
            if frame_count == 0:
                samps = zip(prev_i, cur_i)
                retval = result.raw
                trim_index = out_i * bytes_per_frame - len(retval)
                retval = buffer(retval)[:trim_index]
                return (retval, (d, tuple(samps)))
            for chan in range(nchannels):
                prev_i[chan] = cur_i[chan]
                cur_i[chan] = samples.next()
                cur_i[chan] = (weightA * cur_i[chan] + weightB * prev_i[chan]) / (weightA + weightB)
            frame_count -= 1
            d += outrate
        while d >= 0:
            for chan in range(nchannels):
                cur_o = (prev_i[chan] * d + cur_i[chan] * (outrate - d)) / outrate
                _put_sample(result, size, out_i, _overflow(cur_o, size))
                out_i += 1
            d -= inrate