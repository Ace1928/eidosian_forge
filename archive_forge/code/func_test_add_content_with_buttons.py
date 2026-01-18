import pytest
from kivy.tests.common import GraphicUnitTest
from kivy.base import EventLoop
from kivy.uix.bubble import Bubble
from kivy.uix.bubble import BubbleContent
from kivy.uix.bubble import BubbleButton
def test_add_content_with_buttons(self):
    bubble = Bubble()
    content = BubbleContent()
    content.add_widget(BubbleButton(text='Option A'))
    content.add_widget(BubbleButton(text='Option B'))
    bubble.add_widget(content)
    self.render(bubble)