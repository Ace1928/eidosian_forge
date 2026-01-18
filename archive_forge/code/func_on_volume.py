from jnius import autoclass, java_method, PythonJavaClass
from android import api_version
from kivy.core.audio import Sound, SoundLoader
def on_volume(self, instance, volume):
    if self._mediaplayer:
        volume = float(volume)
        self._mediaplayer.setVolume(volume, volume)