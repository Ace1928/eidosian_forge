from typing import List, Optional
from torchaudio._internal.module_utils import deprecated
from . import utils
from .common import AudioMetaData
def list_audio_backends() -> List[str]:
    """List available backends

    Returns:
        list of str: The list of available backends.

        The possible values are; ``"ffmpeg"``, ``"sox"`` and ``"soundfile"``.
    """
    return list(utils.get_available_backends().keys())