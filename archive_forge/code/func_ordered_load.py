from __future__ import print_function, absolute_import, division
from . import __version__
from .report import report
import os
import importlib
import inspect
import argparse
import distutils.dir_util
import shutil
from collections import OrderedDict
import glob
import sys
import tarfile
import time
import zipfile
import yaml
def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):

    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
    return yaml.load(stream, OrderedLoader)