import importlib
import json
import time
import datetime
import os
import requests
import shutil
import hashlib
import tqdm
import math
import zipfile
import parlai.utils.logging as logging
def download_from_google_drive(gd_id, destination):
    """
    Use the requests package to download a file from Google Drive.
    """
    URL = 'https://docs.google.com/uc?export=download'
    with requests.Session() as session:
        response = session.get(URL, params={'id': gd_id}, stream=True)
        token = _get_confirm_token(response)
        if token:
            response.close()
            params = {'id': gd_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)
        CHUNK_SIZE = 32768
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
        response.close()