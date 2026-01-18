from jnius import autoclass, PythonJavaClass, java_method
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import Fbo, Callback, Rectangle
from kivy.core.camera import CameraBase
import threading
@java_method('([BLandroid/hardware/Camera;)V')
def onPreviewFrame(self, data, camera):
    self._callback(data, camera)