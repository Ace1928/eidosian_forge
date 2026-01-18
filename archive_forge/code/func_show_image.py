from __future__ import annotations
import os
import shutil
import subprocess
import sys
from shlex import quote
from . import Image
def show_image(self, image, **options):
    ipython_display(image)
    return 1