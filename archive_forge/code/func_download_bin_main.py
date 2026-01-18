import argparse
import os
from os import path as op
import shutil
import sys
from . import plugins
from .core import util
def download_bin_main():
    """Argument-parsing wrapper for `download_bin`"""
    description = 'Download plugin binary dependencies'
    phelp = 'Plugin name for which to download the binary. ' + 'If no argument is given, all binaries are downloaded.'
    dhelp = 'Download the binaries to the package directory ' + '(default is the users application data directory). ' + 'This might require administrative rights.'
    example_text = 'examples:\n' + '  imageio_download_bin all\n' + '  imageio_download_bin freeimage\n'
    parser = argparse.ArgumentParser(description=description, epilog=example_text, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('plugin', type=str, nargs='*', default='all', help=phelp)
    parser.add_argument('--package-dir', dest='package_dir', action='store_true', default=False, help=dhelp)
    args = parser.parse_args()
    download_bin(plugin_names=args.plugin, package_dir=args.package_dir)