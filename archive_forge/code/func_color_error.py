import re
from kivy.logger import Logger
from kivy.resources import resource_find
def color_error(text):
    Logger.warning(text)
    return (0, 0, 0, 1)