from pygments import highlight
from pygments import lexers
from pygments import styles
from pygments.formatters import BBCodeFormatter
from kivy.uix.textinput import TextInput
from kivy.core.text.markup import MarkupLabel as Label
from kivy.cache import Cache
from kivy.properties import ObjectProperty, OptionProperty
from kivy.utils import get_hex_from_color, get_color_from_hex
from kivy.uix.behaviors import CodeNavigationBehavior
def on_style_name(self, *args):
    self.style = styles.get_style_by_name(self.style_name)
    self.background_color = get_color_from_hex(self.style.background_color)
    self._trigger_refresh_text()