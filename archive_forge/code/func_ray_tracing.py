import math
from typing import Optional, Tuple, Union
import torch
import torchaudio
from torch import Tensor
def ray_tracing(room: torch.Tensor, source: torch.Tensor, mic_array: torch.Tensor, num_rays: int, absorption: Union[float, torch.Tensor]=0.0, scattering: Union[float, torch.Tensor]=0.0, mic_radius: float=0.5, sound_speed: float=343.0, energy_thres: float=1e-07, time_thres: float=10.0, hist_bin_size: float=0.004) -> torch.Tensor:
    """Compute energy histogram via ray tracing.

    The implementation is based on *pyroomacoustics* :cite:`scheibler2018pyroomacoustics`.

    ``num_rays`` rays are casted uniformly in all directions from the source;
    when a ray intersects a wall, it is reflected and part of its energy is absorbed.
    It is also scattered (sent directly to the microphone(s)) according to the ``scattering``
    coefficient.
    When a ray is close to the microphone, its current energy is recorded in the output
    histogram for that given time slot.

    .. devices:: CPU

    .. properties:: TorchScript

    Args:
        room (torch.Tensor): Room coordinates. The shape of `room` must be `(3,)` which represents
            three dimensions of the room.
        source (torch.Tensor): Sound source coordinates. Tensor with dimensions `(3,)`.
        mic_array (torch.Tensor): Microphone coordinates. Tensor with dimensions `(channel, 3)`.
        absorption (float or torch.Tensor, optional): The absorption coefficients of wall materials.
            (Default: ``0.0``).
            If the type is ``float``, the absorption coefficient is identical to all walls and
            all frequencies.
            If ``absorption`` is a 1D Tensor, the shape must be `(6,)`, representing absorption
            coefficients of ``"west"``, ``"east"``, ``"south"``, ``"north"``, ``"floor"``, and
            ``"ceiling"``, respectively.
            If ``absorption`` is a 2D Tensor, the shape must be  `(num_bands, 6)`.
            ``num_bands`` is the number of frequency bands (usually 7).
        scattering(float or torch.Tensor, optional): The scattering coefficients of wall materials. (Default: ``0.0``)
            The shape and type of this parameter is the same as for ``absorption``.
        mic_radius(float, optional): The radius of the microphone in meters. (Default: 0.5)
        sound_speed (float, optional): The speed of sound in meters per second. (Default: ``343.0``)
        energy_thres (float, optional): The energy level below which we stop tracing a ray. (Default: ``1e-7``)
            The initial energy of each ray is ``2 / num_rays``.
        time_thres (float, optional): The maximal duration for which rays are traced. (Unit: seconds) (Default: 10.0)
        hist_bin_size (float, optional): The size of each bin in the output histogram. (Unit: seconds) (Default: 0.004)

    Returns:
        (torch.Tensor): The 3D histogram(s) where the energy of the traced ray is recorded.
            Each bin corresponds to a given time slot.
            The shape is `(channel, num_bands, num_bins)`, where
            ``num_bins = ceil(time_thres / hist_bin_size)``.
            If both ``absorption`` and ``scattering`` are floats, then ``num_bands == 1``.
    """
    if time_thres < hist_bin_size:
        raise ValueError(f'`time_thres` must be greater than `hist_bin_size`. Found: hist_bin_size={hist_bin_size}, time_thres={time_thres}.')
    if room.dtype != source.dtype or source.dtype != mic_array.dtype:
        raise ValueError(f'dtype of `room`, `source` and `mic_array` must match. Found: `room` ({room.dtype}), `source` ({source.dtype}) and `mic_array` ({mic_array.dtype})')
    _validate_inputs(room, source, mic_array)
    absorption = _adjust_coeff(absorption, 'absorption').to(room.dtype)
    scattering = _adjust_coeff(scattering, 'scattering').to(room.dtype)
    if absorption.shape[0] == 1 and scattering.shape[0] > 1:
        absorption = absorption.expand(scattering.shape)
    if scattering.shape[0] == 1 and absorption.shape[0] > 1:
        scattering = scattering.expand(absorption.shape)
    if absorption.shape != scattering.shape:
        raise ValueError(f'`absorption` and `scattering` must be broadcastable to the same number of bands and walls. Inferred shapes absorption={absorption.shape} and scattering={scattering.shape}')
    histograms = torch.ops.torchaudio.ray_tracing(room, source, mic_array, num_rays, absorption, scattering, mic_radius, sound_speed, energy_thres, time_thres, hist_bin_size)
    return histograms