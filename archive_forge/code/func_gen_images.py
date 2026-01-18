from __future__ import print_function
import argparse
import os
import sys
from importlib import import_module
from jinja2 import Template
from palettable.palette import Palette
def gen_images(palettes, dir_):
    """
    Create images for each palette in the palettes dict.
    For qualitative palettes only the discrete images is made.

    """
    img_dir = os.path.join(dir_, 'img')
    os.makedirs(img_dir, exist_ok=True)
    discrete_fmt = '{}_discrete.png'.format
    continuous_fmt = '{}_continuous.png'.format
    img_size = (6, 0.5)
    for name, p in palettes.items():
        print('Making discrete image for palette {}'.format(name))
        p.save_discrete_image(os.path.join(img_dir, discrete_fmt(name)), size=img_size)
        if p.type != 'qualitative':
            print('Making continuous image for palette {}'.format(name))
            p.save_continuous_image(os.path.join(img_dir, continuous_fmt(name)), size=img_size)