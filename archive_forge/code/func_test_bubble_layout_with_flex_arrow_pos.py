import pytest
from kivy.tests.common import GraphicUnitTest
from kivy.base import EventLoop
from kivy.uix.bubble import Bubble
from kivy.uix.bubble import BubbleContent
from kivy.uix.bubble import BubbleButton
def test_bubble_layout_with_flex_arrow_pos(self):
    for params in bubble_layout_with_flex_arrow_pos_test_params:
        bubble_size = params[:2]
        flex_arrow_pos = params[2:4]
        arrow_side = params[4]
        with self.subTest():
            print('(w={}, h={}, x={}, y={}, side={})'.format(*params))
            bubble = _TestBubble()
            bubble.size_hint = (None, None)
            bubble.size = bubble_size
            bubble.flex_arrow_pos = flex_arrow_pos
            content = _TestBubbleContent(orientation='vertical')
            content.size_hint = (1, 1)
            button = _TestBubbleButton(button_size=(None, None), text='Option')
            button.size_hint_y = 1
            content.add_widget(button)
            bubble.add_widget(content)
            self.render(bubble)
            haw = bubble.arrow_width / 2
            if arrow_side in ['l', 'r']:
                self.assertSequenceAlmostEqual(bubble.arrow_center_pos_within_arrow_layout, (haw, flex_arrow_pos[1]), delta=haw)
            elif arrow_side in ['b', 't']:
                self.assertSequenceAlmostEqual(bubble.arrow_center_pos_within_arrow_layout, (flex_arrow_pos[0], haw), delta=haw)