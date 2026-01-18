from kivy.tests.common import GraphicUnitTest
def test_begin_event_no_dispatch_through_on_touch_events(self):
    win, mouse = self.get_providers(with_window_children=True)
    x, y = win.mouse_pos
    win.dispatch('on_cursor_enter')
    self.advance_frames(1)
    self.assert_event('begin', win.to_normalized_pos(x, y))
    assert self.touch_event is None
    win.dispatch('on_cursor_leave')
    self.advance_frames(1)