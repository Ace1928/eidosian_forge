import os
from setuptools import find_packages
from setuptools import setup
def get_required_install_packages():
    global_names = {}
    with open(os.path.normpath('google/cloud/ml/version.py')) as f:
        exec(f.read(), global_names)
    return global_names['required_install_packages']