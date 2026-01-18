from kivy.tests.common import GraphicUnitTest
def test_multitouch_on_demand_noscatter_righttouch(self):
    self.multitouch_dot_touch('right', on_demand=True)