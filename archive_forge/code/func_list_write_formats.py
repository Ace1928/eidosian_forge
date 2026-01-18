from typing import Dict, List
import torchaudio
def list_write_formats() -> List[str]:
    """List the supported audio formats for write

    Returns:
        List[str]: List of supported audio formats
    """
    return sox_ext.list_write_formats()