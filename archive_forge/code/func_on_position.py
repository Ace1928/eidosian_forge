from json import load
from os.path import exists
from kivy.properties import ObjectProperty, StringProperty, BooleanProperty, \
from kivy.animation import Animation
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.progressbar import ProgressBar
from kivy.uix.label import Label
from kivy.uix.video import Video
from kivy.uix.video import Image
from kivy.factory import Factory
from kivy.logger import Logger
from kivy.clock import Clock
def on_position(self, instance, value):
    labels = self._annotations_labels
    if not labels:
        return
    for label in labels:
        start = label.start
        duration = label.duration
        if start > value or start + duration < value:
            if label.parent:
                label.parent.remove_widget(label)
        elif label.parent is None:
            self.container.add_widget(label)