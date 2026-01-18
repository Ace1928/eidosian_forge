import math
from typing import Optional, Tuple, Union
import torch
import torchaudio
from torch import Tensor
def simulate_rir_ism(room: torch.Tensor, source: torch.Tensor, mic_array: torch.Tensor, max_order: int, absorption: Union[float, torch.Tensor], output_length: Optional[int]=None, delay_filter_length: int=81, center_frequency: Optional[torch.Tensor]=None, sound_speed: float=343.0, sample_rate: float=16000.0) -> Tensor:
    """Compute Room Impulse Response (RIR) based on the *image source method* :cite:`allen1979image`.
    The implementation is based on *pyroomacoustics* :cite:`scheibler2018pyroomacoustics`.

    .. devices:: CPU

    .. properties:: TorchScript

    Args:
        room (torch.Tensor): Room coordinates. The shape of `room` must be `(3,)` which represents
            three dimensions of the room.
        source (torch.Tensor): Sound source coordinates. Tensor with dimensions `(3,)`.
        mic_array (torch.Tensor): Microphone coordinates. Tensor with dimensions `(channel, 3)`.
        max_order (int): The maximum number of reflections of the source.
        absorption (float or torch.Tensor): The *absorption* :cite:`wiki:Absorption_(acoustics)`
            coefficients of wall materials for sound energy.
            If the dtype is ``float``, the absorption coefficient is identical for all walls and
            all frequencies.
            If ``absorption`` is a 1D Tensor, the shape must be `(6,)`, where the values represent
            absorption coefficients of ``"west"``, ``"east"``, ``"south"``, ``"north"``, ``"floor"``,
            and ``"ceiling"``, respectively.
            If ``absorption`` is a 2D Tensor, the shape must be `(7, 6)`, where 7 represents the number of octave bands.
        output_length (int or None, optional): The output length of simulated RIR signal. If ``None``,
            the length is defined as

            .. math::
                \\frac{\\text{max\\_d} \\cdot \\text{sample\\_rate}}{\\text{sound\\_speed}} + \\text{delay\\_filter\\_length}

            where ``max_d`` is the maximum distance between image sources and microphones.
        delay_filter_length (int, optional): The filter length for computing sinc function. (Default: ``81``)
        center_frequency (torch.Tensor, optional): The center frequencies of octave bands for multi-band walls.
            Only used when ``absorption`` is a 2D Tensor.
        sound_speed (float, optional): The speed of sound. (Default: ``343.0``)
        sample_rate (float, optional): The sample rate of the generated room impulse response signal.
            (Default: ``16000.0``)

    Returns:
        (torch.Tensor): The simulated room impulse response waveform. Tensor with dimensions
        `(channel, rir_length)`.

    Note:
        If ``absorption`` is a 2D Tensor and ``center_frequency`` is set to ``None``, the center frequencies
        of octave bands are fixed to ``[125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]``.
        Users need to tune the values of ``absorption`` to the corresponding frequencies.
    """
    _validate_inputs(room, source, mic_array)
    absorption = _adjust_coeff(absorption, 'absorption')
    img_location, att = _compute_image_sources(room, source, max_order, absorption)
    vec = img_location[:, None, :] - mic_array[None, :, :]
    dist = torch.linalg.norm(vec, dim=-1)
    img_src_att = att[..., None] / dist[None, ...]
    delay = dist * sample_rate / sound_speed
    delay_i = torch.ceil(delay)
    irs = img_src_att[..., None] * _frac_delay(delay, delay_i, delay_filter_length)[None, ...]
    rir_length = int(delay_i.max() + irs.shape[-1])
    rir = torch.ops.torchaudio._simulate_rir(irs, delay_i.type(torch.int32), rir_length)
    if absorption.shape[0] > 1:
        if center_frequency is None:
            center = torch.tensor([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0], dtype=room.dtype, device=room.device)
        else:
            center = center_frequency
        filters = torch.ops.torchaudio._make_rir_filter(center, sample_rate, n_fft=512)
        rir = torchaudio.functional.fftconvolve(rir, filters.unsqueeze(1).repeat(1, rir.shape[1], 1), mode='same')
    rir = rir.sum(0)
    if output_length is not None:
        if output_length > rir.shape[-1]:
            rir = torch.nn.functional.pad(rir, (0, output_length - rir.shape[-1]), 'constant', 0.0)
        else:
            rir = rir[..., :output_length]
    return rir