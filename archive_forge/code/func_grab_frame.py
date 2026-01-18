from jnius import autoclass, PythonJavaClass, java_method
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import Fbo, Callback, Rectangle
from kivy.core.camera import CameraBase
import threading
def grab_frame(self):
    """
        Grab current frame (thread-safe, minimal overhead)
        """
    with self._buflock:
        if self._buffer is None:
            return None
        buf = self._buffer.tostring()
        return buf