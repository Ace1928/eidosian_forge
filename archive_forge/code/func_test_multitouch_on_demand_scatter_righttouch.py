from kivy.tests.common import GraphicUnitTest
def test_multitouch_on_demand_scatter_righttouch(self):
    self.multitouch_dot_touch('right', on_demand=True, scatter=True)