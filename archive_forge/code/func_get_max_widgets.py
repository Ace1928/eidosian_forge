from kivy.logger import Logger
from kivy.uix.layout import Layout
from kivy.properties import NumericProperty, BooleanProperty, DictProperty, \
from math import ceil
from itertools import accumulate, product, chain, islice
from operator import sub
def get_max_widgets(self):
    if self.cols and self.rows:
        return self.rows * self.cols
    else:
        return None