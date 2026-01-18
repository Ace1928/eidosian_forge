import json
from struct import pack
from pprint import pprint
from subprocess import Popen
from PIL import Image
from argparse import ArgumentParser
from sys import exit
from os.path import join, exists, dirname, basename
from os import environ, unlink
def locate_texturetool(self):
    search_directories = ['/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/usr/bin/', '/Developer/Platforms/iPhoneOS.platform/Developer/usr/bin/']
    search_directories += environ.get('PATH', '').split(':')
    for directory in search_directories:
        fn = join(directory, 'texturetool')
        if not exists(fn):
            continue
        print('Found texturetool at {}'.format(directory))
        self.texturetool = fn
        return
    print('Error: Unable to locate "texturetool".\nPlease install the iPhone SDK, or the PowerVR SDK.\nThen make sure that "texturetool" is available in your PATH.')
    exit(1)