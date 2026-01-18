import pytest
from kivy.tests.common import GraphicUnitTest
from kivy.base import EventLoop
from kivy.uix.bubble import Bubble
from kivy.uix.bubble import BubbleContent
from kivy.uix.bubble import BubbleButton
def test_add_remove_content(self):
    bubble = Bubble()
    content = BubbleContent()
    bubble.add_widget(content)
    self.render(bubble)
    bubble.remove_widget(content)
    self.render(bubble)