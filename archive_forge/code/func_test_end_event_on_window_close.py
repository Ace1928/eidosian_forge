from kivy.tests.common import GraphicUnitTest
def test_end_event_on_window_close(self):
    win, mouse = self.get_providers()
    x, y = win.mouse_pos = (10.0, 10.0)
    win.dispatch('on_close')
    self.advance_frames(1)
    self.assert_event('end', win.to_normalized_pos(x, y))