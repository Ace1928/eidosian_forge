import os
import re
import sys
import unittest
from collections import defaultdict
from kivy.core.image import ImageLoader
def asset(*fn):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), *fn))