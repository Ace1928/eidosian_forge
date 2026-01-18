import os
import re
import sys
import unittest
from collections import defaultdict
from kivy.core.image import ImageLoader
def test_ImageLoaderTex(self):
    loadercls = LOADERS.get('ImageLoaderTex')
    ctx = self._test_imageloader(loadercls)