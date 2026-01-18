from typing import Dict, List, Tuple
import torio
def get_video_encoders() -> Dict[str, str]:
    """Get the available video encoders.

    Returns:
        Dict[str, str]: Mapping from encoder short name to long name.

    Example
        >>> for k, v in get_audio_encoders().items():
        >>>     print(f"{k}: {v}")
        ... a64multi: Multicolor charset for Commodore 64
        ... a64multi5: Multicolor charset for Commodore 64, extended with 5th color (colram)
        ... alias_pix: Alias/Wavefront PIX image
        ... amv: AMV Video
        ... apng: APNG (Animated Portable Network Graphics) image
        ... asv1: ASUS V1
        ... asv2: ASUS V2
    """
    return ffmpeg_ext.get_video_encoders()