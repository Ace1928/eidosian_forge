from kivy.tests.common import GraphicUnitTest
def test_default_widgets(self):
    from kivy.uix.button import Button
    from kivy.uix.slider import Slider
    r = self.render
    r(Button())
    r(Slider())