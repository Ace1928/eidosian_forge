import json
from struct import pack
from pprint import pprint
from subprocess import Popen
from PIL import Image
from argparse import ArgumentParser
from sys import exit
from os.path import join, exists, dirname, basename
from os import environ, unlink
def runcmd(self, cmd):
    print('Run: {}'.format(' '.join(cmd)))
    Popen(cmd).communicate()