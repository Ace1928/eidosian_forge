import unittest
from tempfile import mkdtemp
from shutil import rmtree
def test_invalid_add_widget(self):
    from kivy.uix.widget import WidgetException
    try:
        self.root.add_widget(None)
        self.root.add_widget(WidgetException)
        self.root.add_widget(self.cls)
        self.fail()
    except WidgetException:
        pass