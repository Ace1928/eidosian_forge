from kivy.tests.common import GraphicUnitTest
def test_no_event_on_cursor_leave(self):
    win, mouse = self.get_providers()
    win.dispatch('on_cursor_leave')
    self.advance_frames(1)
    self.assert_no_event()