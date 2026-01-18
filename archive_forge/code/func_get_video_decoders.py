from typing import Dict, List, Tuple
import torio
def get_video_decoders() -> Dict[str, str]:
    """Get the available video decoders.

    Returns:
        Dict[str, str]: Mapping from decoder short name to long name.

    Example
        >>> for k, v in get_video_decoders().items():
        >>>     print(f"{k}: {v}")
        ... aasc: Autodesk RLE
        ... aic: Apple Intermediate Codec
        ... alias_pix: Alias/Wavefront PIX image
        ... agm: Amuse Graphics Movie
        ... amv: AMV Video
        ... anm: Deluxe Paint Animation
    """
    return ffmpeg_ext.get_video_decoders()