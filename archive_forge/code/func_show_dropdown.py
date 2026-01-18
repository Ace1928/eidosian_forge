from kivy.uix.scrollview import ScrollView
from kivy.properties import ObjectProperty, NumericProperty, BooleanProperty
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.config import Config
def show_dropdown(button, *largs):
    dp = DropDown()
    dp.bind(on_select=lambda instance, x: setattr(button, 'text', x))
    for i in range(10):
        item = Button(text='hello %d' % i, size_hint_y=None, height=44)
        item.bind(on_release=lambda btn: dp.select(btn.text))
        dp.add_widget(item)
    dp.open(button)