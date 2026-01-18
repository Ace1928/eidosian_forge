import pytest
from kivy.tests.common import GraphicUnitTest
from kivy.base import EventLoop
from kivy.uix.bubble import Bubble
from kivy.uix.bubble import BubbleContent
from kivy.uix.bubble import BubbleButton
def assertTestBubbleLayoutWithPredefinedArrowPos(self, bubble):
    arrow_length = bubble.arrow_length
    arrow_width = bubble.arrow_width
    bubble_width = bubble.test_bubble_width
    button_height = bubble.test_button_height
    expected_content_size = {'bottom_left': (bubble_width, button_height), 'bottom_mid': (bubble_width, button_height), 'bottom_right': (bubble_width, button_height), 'top_left': (bubble_width, button_height), 'top_mid': (bubble_width, button_height), 'top_right': (bubble_width, button_height), 'left_top': (bubble_width - arrow_length, button_height), 'left_mid': (bubble_width - arrow_length, button_height), 'left_bottom': (bubble_width - arrow_length, button_height), 'right_top': (bubble_width - arrow_length, button_height), 'right_mid': (bubble_width - arrow_length, button_height), 'right_bottom': (bubble_width - arrow_length, button_height)}[bubble.arrow_pos]
    self.assertSequenceAlmostEqual(bubble.content.size, expected_content_size)
    expected_arrow_layout_size = {'bottom_left': (bubble_width, arrow_length), 'bottom_mid': (bubble_width, arrow_length), 'bottom_right': (bubble_width, arrow_length), 'top_left': (bubble_width, arrow_length), 'top_mid': (bubble_width, arrow_length), 'top_right': (bubble_width, arrow_length), 'left_top': (arrow_length, button_height), 'left_mid': (arrow_length, button_height), 'left_bottom': (arrow_length, button_height), 'right_top': (arrow_length, button_height), 'right_mid': (arrow_length, button_height), 'right_bottom': (arrow_length, button_height)}[bubble.arrow_pos]
    self.assertSequenceAlmostEqual(bubble.arrow_layout_size, expected_arrow_layout_size)
    expected_content_position = {'bottom_left': (0, arrow_length), 'bottom_mid': (0, arrow_length), 'bottom_right': (0, arrow_length), 'top_left': (0, 0), 'top_mid': (0, 0), 'top_right': (0, 0), 'left_top': (arrow_length, 0), 'left_mid': (arrow_length, 0), 'left_bottom': (arrow_length, 0), 'right_top': (0, 0), 'right_mid': (0, 0), 'right_bottom': (0, 0)}[bubble.arrow_pos]
    self.assertSequenceAlmostEqual(bubble.content.pos, expected_content_position)
    expected_arrow_layout_position = {'bottom_left': (0, 0), 'bottom_mid': (0, 0), 'bottom_right': (0, 0), 'top_left': (0, button_height), 'top_mid': (0, button_height), 'top_right': (0, button_height), 'left_top': (0, 0), 'left_mid': (0, 0), 'left_bottom': (0, 0), 'right_top': (bubble_width - arrow_length, 0), 'right_mid': (bubble_width - arrow_length, 0), 'right_bottom': (bubble_width - arrow_length, 0)}[bubble.arrow_pos]
    self.assertSequenceAlmostEqual(bubble.arrow_layout_pos, expected_arrow_layout_position)
    hal = arrow_length / 2
    x_offset = 0.05 * bubble_width
    y_offset = 0.05 * button_height
    expected_arrow_center_pos_within_arrow_layout = {'bottom_left': (x_offset + arrow_width / 2, hal), 'bottom_mid': (bubble_width / 2, hal), 'bottom_right': (bubble_width - arrow_width / 2 - x_offset, hal), 'top_left': (x_offset + arrow_width / 2, hal), 'top_mid': (bubble_width / 2, hal), 'top_right': (bubble_width - arrow_width / 2 - x_offset, hal), 'left_top': (hal, button_height - arrow_width / 2 - y_offset), 'left_mid': (hal, button_height / 2), 'left_bottom': (hal, y_offset + arrow_width / 2), 'right_top': (hal, button_height - arrow_width / 2 - y_offset), 'right_mid': (hal, button_height / 2), 'right_bottom': (hal, y_offset + arrow_width / 2)}[bubble.arrow_pos]
    self.assertSequenceAlmostEqual(bubble.arrow_center_pos_within_arrow_layout, expected_arrow_center_pos_within_arrow_layout)
    expected_arrow_rotation = {'bottom_left': 0, 'bottom_mid': 0, 'bottom_right': 0, 'top_left': 180, 'top_mid': 180, 'top_right': 180, 'left_top': 270, 'left_mid': 270, 'left_bottom': 270, 'right_top': 90, 'right_mid': 90, 'right_bottom': 90}[bubble.arrow_pos]
    self.assertAlmostEqual(bubble.arrow_rotation, expected_arrow_rotation)