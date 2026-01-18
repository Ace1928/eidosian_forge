imported modules that pyinstaller would not find on its own using
import os
import sys
import pkgutil
import logging
from os.path import dirname, join
import importlib
import subprocess
import re
import glob
import kivy
from kivy.factory import Factory
from PyInstaller.depend import bindepend
from os import environ
def get_factory_modules():
    """Returns a list of all the modules registered in the kivy factory.
    """
    mods = [x.get('module', None) for x in Factory.classes.values()]
    return [m for m in mods if m]