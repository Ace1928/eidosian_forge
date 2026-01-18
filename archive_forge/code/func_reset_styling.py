import logging
from functools import partial
from collections import defaultdict
from os.path import splitext
from .diagrams_base import BaseGraph
def reset_styling(self):
    self.custom_styles = {'edge': defaultdict(lambda: defaultdict(str)), 'node': defaultdict(str)}