import pytest
from kivy.tests.common import GraphicUnitTest
from kivy.base import EventLoop
from kivy.uix.bubble import Bubble
from kivy.uix.bubble import BubbleContent
from kivy.uix.bubble import BubbleButton
def test_bubble_layout_without_arrow(self):
    bubble_width = 200
    button_height = 30
    bubble = _TestBubble(show_arrow=False)
    bubble.size_hint = (None, None)

    def update_bubble_size(instance, value):
        w = bubble_width
        h = bubble.content_height
        bubble.size = (w, h)
    bubble.bind(content_size=update_bubble_size)
    content = _TestBubbleContent(orientation='vertical')
    for i in range(7):
        content.add_widget(_TestBubbleButton(button_size=(None, button_height), text='Option_{}'.format(i)))
    bubble.add_widget(content)
    self.render(bubble)
    self.assertSequenceAlmostEqual(bubble.content.size, (bubble_width, 7 * button_height))
    self.assertSequenceAlmostEqual(bubble.content.pos, (0, 0))