import os
import io
import sys
import zipfile
import shutil
from ipywidgets import embed as wembed
import ipyvolume
from ipyvolume.utils import download_to_file, download_to_bytes
from ipyvolume._version import __version_threejs__
def save_ipyvolumejs(target='', devmode=False, version=ipyvolume._version.__version_js__, version3js=__version_threejs__):
    """Output the ipyvolume javascript to a local file.

    :type target: str
    :param bool devmode: if True get index.js from js/dist directory
    :param str version: version number of ipyvolume
    :param str version3js: version number of threejs

    """
    url = 'https://unpkg.com/ipyvolume@{version}/dist/index.js'.format(version=version)
    pyv_filename = 'ipyvolume_v{version}.js'.format(version=version)
    pyv_filepath = os.path.join(target, pyv_filename)
    devfile = os.path.join(os.path.abspath(ipyvolume.__path__[0]), '..', 'js', 'dist', 'index.js')
    devfile_share = os.path.join(sys.prefix, 'share/jupyter/nbextensions/ipyvolume/index.js')
    if devmode:
        if target and (not os.path.exists(target)):
            os.makedirs(target)
        if os.path.exists(devfile_share):
            shutil.copy(devfile_share, pyv_filepath)
        else:
            if not os.path.exists(devfile):
                raise IOError(f'devmode=True but cannot find : {devfile} or {devfile_share}')
            shutil.copy(devfile, pyv_filepath)
    else:
        download_to_file(url, pyv_filepath)
    return pyv_filename