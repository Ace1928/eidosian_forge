from typing import Dict, List
import torchaudio
def get_buffer_size() -> int:
    """Get buffer size for sox effect chain

    Returns:
        int: size in bytes of buffers used for processing audio.
    """
    return sox_ext.get_buffer_size()