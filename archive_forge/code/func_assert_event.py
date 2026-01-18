from kivy.tests.common import GraphicUnitTest
def assert_event(self, etype, spos):
    assert self.etype == etype
    assert 'pos' in self.motion_event.profile
    assert self.motion_event.is_touch is False
    assert self.motion_event.spos == spos