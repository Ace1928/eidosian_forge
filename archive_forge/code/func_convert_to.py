import sys
from PIL import Image
from os.path import exists, join, realpath, basename, dirname
from os import makedirs
from argparse import ArgumentParser
def convert_to(self, source, icon_fn, size):
    dest = source.resize((size, size))
    dest.save(icon_fn, 'png')