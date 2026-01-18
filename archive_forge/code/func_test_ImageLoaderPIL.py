import os
import re
import sys
import unittest
from collections import defaultdict
from kivy.core.image import ImageLoader
def test_ImageLoaderPIL(self):
    loadercls = LOADERS.get('ImageLoaderPIL')
    ctx = self._test_imageloader(loadercls)