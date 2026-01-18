from scipy.signal import butter, sosfilt
from .utils import (register_pydub_effect,stereo_to_ms,ms_to_stereo)
@register_pydub_effect
def high_pass_filter(seg, cutoff_freq, order=5):
    filter_fn = _mk_butter_filter(cutoff_freq, 'highpass', order=order)
    return seg.apply_mono_filter_to_each_channel(filter_fn)