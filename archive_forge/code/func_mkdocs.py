from __future__ import print_function
import argparse
import os
import sys
from importlib import import_module
from jinja2 import Template
from palettable.palette import Palette
def mkdocs(mod, dir_, images=False):
    palettes = find_palettes(mod)
    if images:
        gen_images(palettes, dir_)
    render_doc_page(dir_, sorted(palettes.keys(), key=palette_name_sort_key), palettes)