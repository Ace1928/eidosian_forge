import pytest
from kivy.tests.common import GraphicUnitTest
from kivy.base import EventLoop
from kivy.uix.bubble import Bubble
from kivy.uix.bubble import BubbleContent
from kivy.uix.bubble import BubbleButton
def test_add_two_content_widgets_fails(self):
    from kivy.uix.bubble import BubbleException
    bubble = Bubble()
    content_1 = BubbleContent()
    content_2 = BubbleContent()
    bubble.add_widget(content_1)
    with self.assertRaises(BubbleException):
        bubble.add_widget(content_2)