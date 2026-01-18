import pytest
from kivy.tests.common import GraphicUnitTest
from kivy.base import EventLoop
from kivy.uix.bubble import Bubble
from kivy.uix.bubble import BubbleContent
from kivy.uix.bubble import BubbleButton
def test_bubble_layout_with_predefined_arrow_pos(self):
    for params in bubble_layout_with_predefined_arrow_pos_test_params:
        bubble_width, button_height, arrow_pos = params
        with self.subTest():
            print('(bubble_width={}, button_height={}, arrow_pos={})'.format(*params))
            bubble = _TestBubble(arrow_pos=arrow_pos)
            bubble.size_hint = (None, None)
            bubble.test_bubble_width = bubble_width
            bubble.test_button_height = button_height

            def update_bubble_size(instance, value):
                w = bubble_width
                h = bubble.content_height + bubble.arrow_margin_y
                bubble.size = (w, h)
            bubble.bind(content_size=update_bubble_size, arrow_margin=update_bubble_size)
            content = _TestBubbleContent()
            for i in range(3):
                content.add_widget(_TestBubbleButton(button_size=(None, button_height), text='Option {}'.format(i)))
            bubble.add_widget(content)
            self.render(bubble)
            self.assertTestBubbleLayoutWithPredefinedArrowPos(bubble)