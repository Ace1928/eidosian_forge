from kivy.uix.image import Image
from kivy.uix.scatter import Scatter
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.button import Button
from kivy.properties import ObjectProperty
from kivy.properties import StringProperty
from kivy.properties import OptionProperty
from kivy.properties import ListProperty
from kivy.properties import BooleanProperty
from kivy.properties import ColorProperty
from kivy.properties import NumericProperty
from kivy.properties import ReferenceListProperty
from kivy.base import EventLoop
from kivy.metrics import dp
def on_arrow_image(self, instance, value):
    self._arrow_image.source = self.arrow_image
    self._arrow_image.width = self._arrow_image.texture_size[0]
    self._arrow_image.height = dp(self._arrow_image.texture_size[1])
    self._arrow_image_scatter.size = self._arrow_image.texture_size
    self.reposition_inner_widgets()