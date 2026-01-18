from itertools import product
from kivy.tests import GraphicUnitTest
from kivy.logger import LoggerHistory
def check_opacity_support(self):
    LoggerHistory.clear_history()
    self.Window.opacity = self.get_new_opacity_value()
    return not LoggerHistory.history