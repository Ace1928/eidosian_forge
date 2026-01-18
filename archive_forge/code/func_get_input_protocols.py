from typing import Dict, List, Tuple
import torio
def get_input_protocols() -> List[str]:
    """Get the supported input protocols.

    Returns:
        List[str]: The names of supported input protocols

    Example
        >>> print(get_input_protocols())
        ... ['file', 'ftp', 'hls', 'http','https', 'pipe', 'rtmp', 'tcp', 'tls', 'udp', 'unix']
    """
    return ffmpeg_ext.get_input_protocols()