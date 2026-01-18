import sys
import os.path
import pkgutil
import shutil
import tempfile
import argparse
import importlib
from base64 import b85decode
def include_setuptools(args):
    """
    Install setuptools only if absent and not excluded.
    """
    cli = not args.no_setuptools
    env = not os.environ.get('PIP_NO_SETUPTOOLS')
    absent = not importlib.util.find_spec('setuptools')
    return cli and env and absent