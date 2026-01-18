import os
import io
import sys
import zipfile
import shutil
from ipywidgets import embed as wembed
import ipyvolume
from ipyvolume.utils import download_to_file, download_to_bytes
from ipyvolume._version import __version_threejs__
def save_font_awesome(dirpath='', version='4.7.0'):
    """Download and save the font-awesome package to a local directory.

    :type dirpath: str
    :type url: str

    """
    directory_name = 'font-awesome-{0:s}'.format(version)
    directory_path = os.path.join(dirpath, directory_name)
    if os.path.exists(directory_path):
        return directory_name
    url = 'https://fontawesome.com/v{0:s}/assets/font-awesome-{0:s}.zip'.format(version)
    content, _encoding = download_to_bytes(url)
    try:
        zip_directory = io.BytesIO(content)
        unzip = zipfile.ZipFile(zip_directory)
        top_level_name = unzip.namelist()[0]
        unzip.extractall(dirpath)
    except Exception as err:
        raise IOError('Could not unzip content from: {0}\n{1}'.format(url, err))
    os.rename(os.path.join(dirpath, top_level_name), directory_path)
    return directory_name