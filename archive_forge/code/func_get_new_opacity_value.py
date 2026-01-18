from itertools import product
from kivy.tests import GraphicUnitTest
from kivy.logger import LoggerHistory
def get_new_opacity_value(self):
    opacity = self.Window.opacity
    opacity = opacity - 0.1 if opacity >= 0.9 else opacity + 0.1
    return round(opacity, 2)