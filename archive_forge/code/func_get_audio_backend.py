from typing import List, Optional
from torchaudio._internal.module_utils import deprecated
from . import utils
from .common import AudioMetaData
@deprecated('With dispatcher enabled, this function is no-op. You can remove the function call.')
def get_audio_backend() -> Optional[str]:
    """Get the name of the current global backend

    Returns:
        str or None:
            If dispatcher mode is enabled, returns ``None`` otherwise,
            the name of current backend or ``None`` (no backend is set).
    """
    return None