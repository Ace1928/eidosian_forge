from kivy.tests.common import GraphicUnitTest
def multitouch_dot_move(self, button, **kwargs):
    eventloop, win, mouse, wid = self.mouse_init(**kwargs)
    mouse.start()
    eventloop.add_input_provider(mouse)
    self.assertEqual(mouse.counter, 0)
    self.assertEqual(mouse.touches, {})
    win.dispatch('on_mouse_down', 10, self.correct_y(win, 10), 'right', {})
    event_id = next(iter(mouse.touches))
    self.assertEqual(mouse.counter, 1)
    if 'on_demand' in kwargs and 'scatter' not in kwargs:
        self.render(wid)
        mouse.stop()
        eventloop.remove_input_provider(mouse)
        return
    elif 'on_demand' in kwargs and 'scatter' in kwargs:
        self.assertIn('multitouch_sim', mouse.touches[event_id].profile)
        self.assertTrue(mouse.multitouch_on_demand)
        self.advance_frames(1)
        wid.on_touch_down(mouse.touches[event_id])
        wid.on_touch_up(mouse.touches[event_id])
        self.assertTrue(mouse.touches[event_id].multitouch_sim)
        win.dispatch('on_mouse_up', 10, self.correct_y(win, 10), 'right', {})
        ellipse = mouse.touches[event_id].ud.get('_drawelement')[1].proxy_ref
        win.dispatch('on_mouse_down', 10, self.correct_y(win, 10), 'right', {})
    elif 'disabled' in kwargs:
        self.assertIsNone(mouse.touches[event_id].ud.get('_drawelement'))
    else:
        self.assertIsNotNone(mouse.touches[event_id].ud.get('_drawelement'))
    if 'disabled' in kwargs:
        self.assertIsNone(mouse.touches[event_id].ud.get('_drawelement'))
        mouse.stop()
        eventloop.remove_input_provider(mouse)
        return
    else:
        ellipse = mouse.touches[event_id].ud.get('_drawelement')[1].proxy_ref
    win.dispatch('on_mouse_move', 11, self.correct_y(win, 11), {})
    self.assertEqual(ellipse.pos, (1, 1))
    win.dispatch('on_mouse_up', 10, self.correct_y(win, 10), 'right', {})
    self.assertEqual(mouse.counter, 1)
    self.assertIn(event_id, mouse.touches)
    self.assertIsNotNone(mouse.touches[event_id].ud.get('_drawelement'))
    win.dispatch('on_mouse_down', 10, self.correct_y(win, 10), button, {})
    self.assertEqual(mouse.counter, 1)
    self.assertIsNotNone(mouse.touches[event_id].ud.get('_drawelement'))
    win.dispatch('on_mouse_move', 50, self.correct_y(win, 50), {})
    self.assertEqual(ellipse.pos, (40, 40))
    win.dispatch('on_mouse_up', 10, self.correct_y(win, 10), button, {})
    self.assertEqual(mouse.counter, 1)
    if button == 'left':
        self.assertNotIn(event_id, mouse.touches)
    elif button == 'right':
        self.assertIn(event_id, mouse.touches)
        self.assertIsNotNone(mouse.touches[event_id].ud.get('_drawelement'))
    self.render(wid)
    mouse.stop()
    eventloop.remove_input_provider(mouse)