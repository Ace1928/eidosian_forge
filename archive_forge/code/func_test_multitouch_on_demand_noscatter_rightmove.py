from kivy.tests.common import GraphicUnitTest
def test_multitouch_on_demand_noscatter_rightmove(self):
    self.multitouch_dot_move('right', on_demand=True)