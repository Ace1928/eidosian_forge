from kivy.tests.common import GraphicUnitTest
def assert_no_event(self):
    assert self.etype is None
    assert self.motion_event is None