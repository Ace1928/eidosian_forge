import math
import warnings
from typing import Optional
import torch
from torch import Tensor
from torchaudio._extension import _IS_TORCHAUDIO_EXT_AVAILABLE
def flanger(waveform: Tensor, sample_rate: int, delay: float=0.0, depth: float=2.0, regen: float=0.0, width: float=71.0, speed: float=0.5, phase: float=25.0, modulation: str='sinusoidal', interpolation: str='linear') -> Tensor:
    """Apply a flanger effect to the audio. Similar to SoX implementation.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., channel, time)` .
            Max 4 channels allowed
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
        delay (float, optional): desired delay in milliseconds(ms)
            Allowed range of values are 0 to 30
        depth (float, optional): desired delay depth in milliseconds(ms)
            Allowed range of values are 0 to 10
        regen (float, optional): desired regen(feedback gain) in dB
            Allowed range of values are -95 to 95
        width (float, optional):  desired width(delay gain) in dB
            Allowed range of values are 0 to 100
        speed (float, optional):  modulation speed in Hz
            Allowed range of values are 0.1 to 10
        phase (float, optional):  percentage phase-shift for multi-channel
            Allowed range of values are 0 to 100
        modulation (str, optional):  Use either "sinusoidal" or "triangular" modulation. (Default: ``sinusoidal``)
        interpolation (str, optional): Use either "linear" or "quadratic" for delay-line interpolation.
            (Default: ``linear``)

    Returns:
        Tensor: Waveform of dimension of `(..., channel, time)`

    Reference:
        - http://sox.sourceforge.net/sox.html

        - Scott Lehman, `Effects Explained`_,

    .. _Effects Explained:
        https://web.archive.org/web/20051125072557/http://www.harmony-central.com/Effects/effects-explained.html
    """
    if modulation not in ('sinusoidal', 'triangular'):
        raise ValueError('Only "sinusoidal" or "triangular" modulation allowed')
    if interpolation not in ('linear', 'quadratic'):
        raise ValueError('Only "linear" or "quadratic" interpolation allowed')
    actual_shape = waveform.shape
    device, dtype = (waveform.device, waveform.dtype)
    if actual_shape[-2] > 4:
        raise ValueError('Max 4 channels allowed')
    waveform = waveform.view(-1, actual_shape[-2], actual_shape[-1])
    feedback_gain = regen / 100
    delay_gain = width / 100
    channel_phase = phase / 100
    delay_min = delay / 1000
    delay_depth = depth / 1000
    n_channels = waveform.shape[-2]
    if modulation == 'sinusoidal':
        wave_type = 'SINE'
    else:
        wave_type = 'TRIANGLE'
    in_gain = 1.0 / (1 + delay_gain)
    delay_gain = delay_gain / (1 + delay_gain)
    delay_gain = delay_gain * (1 - abs(feedback_gain))
    delay_buf_length = int((delay_min + delay_depth) * sample_rate + 0.5)
    delay_buf_length = delay_buf_length + 2
    delay_bufs = torch.zeros(waveform.shape[0], n_channels, delay_buf_length, dtype=dtype, device=device)
    delay_last = torch.zeros(waveform.shape[0], n_channels, dtype=dtype, device=device)
    lfo_length = int(sample_rate / speed)
    table_min = math.floor(delay_min * sample_rate + 0.5)
    table_max = delay_buf_length - 2.0
    lfo = _generate_wave_table(wave_type=wave_type, data_type='FLOAT', table_size=lfo_length, min=float(table_min), max=float(table_max), phase=3 * math.pi / 2, device=device)
    output_waveform = torch.zeros_like(waveform, dtype=dtype, device=device)
    delay_buf_pos = 0
    lfo_pos = 0
    channel_idxs = torch.arange(0, n_channels, device=device)
    for i in range(waveform.shape[-1]):
        delay_buf_pos = (delay_buf_pos + delay_buf_length - 1) % delay_buf_length
        cur_channel_phase = (channel_idxs * lfo_length * channel_phase + 0.5).to(torch.int64)
        delay_tensor = lfo[(lfo_pos + cur_channel_phase) % lfo_length]
        frac_delay = torch.frac(delay_tensor)
        delay_tensor = torch.floor(delay_tensor)
        int_delay = delay_tensor.to(torch.int64)
        temp = waveform[:, :, i]
        delay_bufs[:, :, delay_buf_pos] = temp + delay_last * feedback_gain
        delayed_0 = delay_bufs[:, channel_idxs, (delay_buf_pos + int_delay) % delay_buf_length]
        int_delay = int_delay + 1
        delayed_1 = delay_bufs[:, channel_idxs, (delay_buf_pos + int_delay) % delay_buf_length]
        int_delay = int_delay + 1
        if interpolation == 'linear':
            delayed = delayed_0 + (delayed_1 - delayed_0) * frac_delay
        else:
            delayed_2 = delay_bufs[:, channel_idxs, (delay_buf_pos + int_delay) % delay_buf_length]
            int_delay = int_delay + 1
            delayed_2 = delayed_2 - delayed_0
            delayed_1 = delayed_1 - delayed_0
            a = delayed_2 * 0.5 - delayed_1
            b = delayed_1 * 2 - delayed_2 * 0.5
            delayed = delayed_0 + (a * frac_delay + b) * frac_delay
        delay_last = delayed
        output_waveform[:, :, i] = waveform[:, :, i] * in_gain + delayed * delay_gain
        lfo_pos = (lfo_pos + 1) % lfo_length
    return output_waveform.clamp(min=-1, max=1).view(actual_shape)