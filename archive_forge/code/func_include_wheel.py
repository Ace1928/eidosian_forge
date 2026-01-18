import sys
import os.path
import pkgutil
import shutil
import tempfile
import argparse
import importlib
from base64 import b85decode
def include_wheel(args):
    """
    Install wheel only if absent and not excluded.
    """
    cli = not args.no_wheel
    env = not os.environ.get('PIP_NO_WHEEL')
    absent = not importlib.util.find_spec('wheel')
    return cli and env and absent