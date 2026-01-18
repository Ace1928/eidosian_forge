import warnings
from typing import Optional, Union
import numpy as np
def mel_to_hertz(mels: Union[float, np.ndarray], mel_scale: str='htk') -> Union[float, np.ndarray]:
    """
    Convert frequency from mels to hertz.

    Args:
        mels (`float` or `np.ndarray`):
            The frequency, or multiple frequencies, in mels.
        mel_scale (`str`, *optional*, `"htk"`):
            The mel frequency scale to use, `"htk"`, `"kaldi"` or `"slaney"`.

    Returns:
        `float` or `np.ndarray`: The frequencies in hertz.
    """
    if mel_scale not in ['slaney', 'htk', 'kaldi']:
        raise ValueError('mel_scale should be one of "htk", "slaney" or "kaldi".')
    if mel_scale == 'htk':
        return 700.0 * (np.power(10, mels / 2595.0) - 1.0)
    elif mel_scale == 'kaldi':
        return 700.0 * (np.exp(mels / 1127.0) - 1.0)
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0
    if isinstance(mels, np.ndarray):
        log_region = mels >= min_log_mel
        freq[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
    elif mels >= min_log_mel:
        freq = min_log_hertz * np.exp(logstep * (mels - min_log_mel))
    return freq