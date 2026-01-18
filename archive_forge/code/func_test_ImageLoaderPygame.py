import os
import re
import sys
import unittest
from collections import defaultdict
from kivy.core.image import ImageLoader
def test_ImageLoaderPygame(self):
    loadercls = LOADERS.get('ImageLoaderPygame')
    ctx = self._test_imageloader(loadercls)