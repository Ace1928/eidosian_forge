import os
import re
import sys
import unittest
from collections import defaultdict
from kivy.core.image import ImageLoader
def test_ImageLoaderGIF(self):
    loadercls = LOADERS.get('ImageLoaderGIF')
    ctx = self._test_imageloader(loadercls)