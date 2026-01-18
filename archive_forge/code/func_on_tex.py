from kivy.uix.image import Image
from kivy.core.camera import Camera as CoreCamera
from kivy.properties import NumericProperty, ListProperty, \
def on_tex(self, camera):
    self.texture = texture = camera.texture
    self.texture_size = list(texture.size)
    self.canvas.ask_update()