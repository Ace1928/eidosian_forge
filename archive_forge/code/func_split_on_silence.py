import itertools
from .utils import db_to_float
def split_on_silence(audio_segment, min_silence_len=1000, silence_thresh=-16, keep_silence=100, seek_step=1):
    """
    Returns list of audio segments from splitting audio_segment on silent sections

    audio_segment - original pydub.AudioSegment() object

    min_silence_len - (in ms) minimum length of a silence to be used for
        a split. default: 1000ms

    silence_thresh - (in dBFS) anything quieter than this will be
        considered silence. default: -16dBFS

    keep_silence - (in ms or True/False) leave some silence at the beginning
        and end of the chunks. Keeps the sound from sounding like it
        is abruptly cut off.
        When the length of the silence is less than the keep_silence duration
        it is split evenly between the preceding and following non-silent
        segments.
        If True is specified, all the silence is kept, if False none is kept.
        default: 100ms

    seek_step - step size for interating over the segment in ms
    """

    def pairwise(iterable):
        """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)
    if isinstance(keep_silence, bool):
        keep_silence = len(audio_segment) if keep_silence else 0
    output_ranges = [[start - keep_silence, end + keep_silence] for start, end in detect_nonsilent(audio_segment, min_silence_len, silence_thresh, seek_step)]
    for range_i, range_ii in pairwise(output_ranges):
        last_end = range_i[1]
        next_start = range_ii[0]
        if next_start < last_end:
            range_i[1] = (last_end + next_start) // 2
            range_ii[0] = range_i[1]
    return [audio_segment[max(start, 0):min(end, len(audio_segment))] for start, end in output_ranges]