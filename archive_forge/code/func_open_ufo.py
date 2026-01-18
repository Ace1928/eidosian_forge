import os
import argparse
import logging
import shutil
import multiprocessing as mp
from contextlib import closing
from functools import partial
import fontTools
from .ufo import font_to_quadratic, fonts_to_quadratic
def open_ufo(path):
    if hasattr(ufo_module.Font, 'open'):
        return ufo_module.Font.open(path)
    return ufo_module.Font(path)