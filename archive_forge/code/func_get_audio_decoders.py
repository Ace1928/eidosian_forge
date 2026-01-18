from typing import Dict, List, Tuple
import torio
def get_audio_decoders() -> Dict[str, str]:
    """Get the available audio decoders.

    Returns:
        Dict[str, str]: Mapping from decoder short name to long name.

    Example
        >>> for k, v in get_audio_decoders().items():
        >>>     print(f"{k}: {v}")
        ... a64: a64 - video for Commodore 64
        ... ac3: raw AC-3
        ... adts: ADTS AAC (Advanced Audio Coding)
        ... adx: CRI ADX
        ... aiff: Audio IFF
    """
    return ffmpeg_ext.get_audio_decoders()