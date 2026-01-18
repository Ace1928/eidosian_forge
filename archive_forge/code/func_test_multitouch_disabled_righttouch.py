from kivy.tests.common import GraphicUnitTest
def test_multitouch_disabled_righttouch(self):
    self.multitouch_dot_touch('right', disabled=True)