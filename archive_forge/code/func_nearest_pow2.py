import json
from struct import pack
from pprint import pprint
from subprocess import Popen
from PIL import Image
from argparse import ArgumentParser
from sys import exit
from os.path import join, exists, dirname, basename
from os import environ, unlink
def nearest_pow2(self, v):
    v -= 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    return v + 1