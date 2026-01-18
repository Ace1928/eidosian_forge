import sys
import os.path
import pkgutil
import shutil
import tempfile
import argparse
import importlib
from base64 import b85decode
def determine_pip_install_arguments():
    pre_parser = argparse.ArgumentParser()
    pre_parser.add_argument('--no-setuptools', action='store_true')
    pre_parser.add_argument('--no-wheel', action='store_true')
    pre, args = pre_parser.parse_known_args()
    args.append('pip')
    if include_setuptools(pre):
        args.append('setuptools')
    if include_wheel(pre):
        args.append('wheel')
    return ['install', '--upgrade', '--force-reinstall'] + args