import os
import re
import sys
import unittest
from collections import defaultdict
from kivy.core.image import ImageLoader
def has_alpha(fmt):
    return fmt in ('rgba', 'bgra', 'argb', 'abgr')