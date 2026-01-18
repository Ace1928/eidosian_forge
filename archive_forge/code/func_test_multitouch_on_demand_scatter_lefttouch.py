from kivy.tests.common import GraphicUnitTest
def test_multitouch_on_demand_scatter_lefttouch(self):
    self.multitouch_dot_touch('left', on_demand=True, scatter=True)