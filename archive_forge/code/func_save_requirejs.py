import os
import io
import sys
import zipfile
import shutil
from ipywidgets import embed as wembed
import ipyvolume
from ipyvolume.utils import download_to_file, download_to_bytes
from ipyvolume._version import __version_threejs__
def save_requirejs(target='', version='2.3.4'):
    """Download and save the require javascript to a local file.

    :type target: str
    :type version: str
    """
    url = 'https://cdnjs.cloudflare.com/ajax/libs/require.js/{version}/require.min.js'.format(version=version)
    filename = 'require.min.v{0}.js'.format(version)
    filepath = os.path.join(target, filename)
    download_to_file(url, filepath)
    return filename