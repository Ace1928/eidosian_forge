import sys
import math
import array
from .utils import (
from .silence import split_on_silence
from .exceptions import TooManyMissingFrames, InvalidDuration
@register_pydub_effect
def compress_dynamic_range(seg, threshold=-20.0, ratio=4.0, attack=5.0, release=50.0):
    """
    Keyword Arguments:
        
        threshold - default: -20.0
            Threshold in dBFS. default of -20.0 means -20dB relative to the
            maximum possible volume. 0dBFS is the maximum possible value so
            all values for this argument sould be negative.

        ratio - default: 4.0
            Compression ratio. Audio louder than the threshold will be 
            reduced to 1/ratio the volume. A ratio of 4.0 is equivalent to
            a setting of 4:1 in a pro-audio compressor like the Waves C1.
        
        attack - default: 5.0
            Attack in milliseconds. How long it should take for the compressor
            to kick in once the audio has exceeded the threshold.

        release - default: 50.0
            Release in milliseconds. How long it should take for the compressor
            to stop compressing after the audio has falled below the threshold.

    
    For an overview of Dynamic Range Compression, and more detailed explanation
    of the related terminology, see: 

        http://en.wikipedia.org/wiki/Dynamic_range_compression
    """
    thresh_rms = seg.max_possible_amplitude * db_to_float(threshold)
    look_frames = int(seg.frame_count(ms=attack))

    def rms_at(frame_i):
        return seg.get_sample_slice(frame_i - look_frames, frame_i).rms

    def db_over_threshold(rms):
        if rms == 0:
            return 0.0
        db = ratio_to_db(rms / thresh_rms)
        return max(db, 0)
    output = []
    attenuation = 0.0
    attack_frames = seg.frame_count(ms=attack)
    release_frames = seg.frame_count(ms=release)
    for i in xrange(int(seg.frame_count())):
        rms_now = rms_at(i)
        max_attenuation = (1 - 1.0 / ratio) * db_over_threshold(rms_now)
        attenuation_inc = max_attenuation / attack_frames
        attenuation_dec = max_attenuation / release_frames
        if rms_now > thresh_rms and attenuation <= max_attenuation:
            attenuation += attenuation_inc
            attenuation = min(attenuation, max_attenuation)
        else:
            attenuation -= attenuation_dec
            attenuation = max(attenuation, 0)
        frame = seg.get_frame(i)
        if attenuation != 0.0:
            frame = audioop.mul(frame, seg.sample_width, db_to_float(-attenuation))
        output.append(frame)
    return seg._spawn(data=b''.join(output))