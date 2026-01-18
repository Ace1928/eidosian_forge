from kivy.uix.image import Image
from kivy.uix.scatter import Scatter
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.button import Button
from kivy.properties import ObjectProperty
from kivy.properties import StringProperty
from kivy.properties import OptionProperty
from kivy.properties import ListProperty
from kivy.properties import BooleanProperty
from kivy.properties import ColorProperty
from kivy.properties import NumericProperty
from kivy.properties import ReferenceListProperty
from kivy.base import EventLoop
from kivy.metrics import dp
def reposition_inner_widgets(self):
    arrow_image_layout = self._arrow_image_layout
    arrow_image_scatter = self._arrow_image_scatter
    arrow_image_scatter_wrapper = self._arrow_image_scatter_wrapper
    content = self.content
    for child in list(self.children):
        super().remove_widget(child)
    if self.canvas is None or content is None:
        return
    if self._flex_arrow_layout_params is not None:
        layout_params = self._flex_arrow_layout_params
    else:
        layout_params = Bubble.ARROW_LAYOUTS[self.arrow_pos]
    bubble_orientation, widget_order, arrow_size_hint, arrow_rotation, arrow_pos_hint = layout_params
    arrow_image_scatter.rotation = arrow_rotation
    arrow_image_scatter_wrapper.size = arrow_image_scatter.bbox[1]
    arrow_image_scatter_wrapper.pos_hint = arrow_pos_hint
    arrow_image_layout.size_hint = arrow_size_hint
    arrow_image_layout.size = arrow_image_scatter.bbox[1]
    self.orientation = bubble_orientation
    widgets_to_add = [content, arrow_image_layout]
    arrow_margin_x, arrow_margin_y = (0, 0)
    if self.show_arrow:
        if bubble_orientation[0] == 'h':
            arrow_margin_x = arrow_image_layout.width
        elif bubble_orientation[0] == 'v':
            arrow_margin_y = arrow_image_layout.height
    else:
        widgets_to_add.pop(1)
    for widget in widgets_to_add[::widget_order]:
        super().add_widget(widget)
    self.arrow_margin = (arrow_margin_x, arrow_margin_y)