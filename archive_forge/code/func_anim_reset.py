import re
from base64 import b64decode
import imghdr
from kivy.event import EventDispatcher
from kivy.core import core_register_libs
from kivy.logger import Logger
from kivy.cache import Cache
from kivy.clock import Clock
from kivy.atlas import Atlas
from kivy.resources import resource_find
from kivy.utils import platform
from kivy.compat import string_types
from kivy.setupconfig import USE_SDL2
import zipfile
from io import BytesIO
from os import environ
from kivy.graphics.texture import Texture, TextureRegion
def anim_reset(self, allow_anim):
    """Reset an animation if available.

        .. versionadded:: 1.0.8

        :Parameters:
            `allow_anim`: bool
                Indicate whether the animation should restart playing or not.

        Usage::

            # start/reset animation
            image.anim_reset(True)

            # or stop the animation
            image.anim_reset(False)

        You can change the animation speed whilst it is playing::

            # Set to 20 FPS
            image.anim_delay = 1 / 20.

        """
    if self._anim_ev is not None:
        self._anim_ev.cancel()
        self._anim_ev = None
    if allow_anim and self._anim_available and (self._anim_delay >= 0):
        self._anim_ev = Clock.schedule_interval(self._anim, self.anim_delay)
        self._anim()