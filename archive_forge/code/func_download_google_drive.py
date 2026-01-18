from .. import utils
from .._lazyload import requests
import os
import tempfile
import urllib.request
import zipfile
def download_google_drive(id, destination):
    """Download a file from Google Drive.

    Requires the file to be available to view by anyone with the URL.

    Parameters
    ----------
    id : string
        Google Drive ID string. You can access this by clicking 'Get Shareable Link',
        which will give a URL of the form
        <https://drive.google.com/file/d/**your_file_id**/view?usp=sharing>
    destination : string or file
        File to which to save the downloaded data
    """
    response = _GET_google_drive(id)
    _save_response_content(response, destination)