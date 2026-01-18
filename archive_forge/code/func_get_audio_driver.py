import sys
import atexit
import pyglet
def get_audio_driver():
    """Get the preferred audio driver for the current platform.

    See :data:`pyglet.options` ``audio``, and the Programming guide,
    section :doc:`/programming_guide/media` for more information on
    setting the preferred driver.

    Returns:
        AbstractAudioDriver : The concrete implementation of the preferred
                              audio driver for this platform.
    """
    return _audio_driver