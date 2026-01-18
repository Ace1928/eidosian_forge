import warnings
from pyglet.util import CodecRegistry, Decoder, Encoder
from .base import *
import pyglet
def have_ffmpeg():
    """Check if FFmpeg library is available.

    Returns:
        bool: True if FFmpeg is found.

    .. versionadded:: 1.4
    """
    try:
        from . import ffmpeg_lib
        if _debug:
            print('FFmpeg available, using to load media files.')
        found = False
        for release_versions, build_versions in ffmpeg_lib.release_versions.items():
            if ffmpeg_lib.compat.versions == build_versions:
                if _debug:
                    print(f'FFMpeg Release Version: {release_versions}. Versions Loaded: {ffmpeg_lib.compat.versions}')
                found = True
                break
        if not found:
            warnings.warn(f'FFmpeg release version not found. This may be a new or untested release. Unknown behavior may occur. Versions Loaded: {ffmpeg_lib.compat.versions}')
        return True
    except (ImportError, FileNotFoundError, AttributeError) as err:
        if _debug:
            print(f'FFmpeg not available: {err}')
        return False