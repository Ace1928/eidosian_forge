from jnius import autoclass, java_method, PythonJavaClass
from android import api_version
from kivy.core.audio import Sound, SoundLoader
@java_method('(Landroid/media/MediaPlayer;)V')
def onCompletion(self, mp):
    self.callback()