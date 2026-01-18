import sys
from PIL import Image
from os.path import exists, join, realpath, basename, dirname
from os import makedirs
from argparse import ArgumentParser
def ensure_quality(self, image, force=False):
    messages = []
    w, h = image.size
    if w != h:
        messages.append('Width and height should be the same')
    if w not in (512, 1024):
        messages.append('Source image is recommended to be 1024 (512 minimum)')
    if not messages:
        return
    print('Quality check failed')
    for message in messages:
        print('- {}'.format(message))
    if not force:
        sys.exit(1)