import re
import os.path
from typing import Dict, Union, Optional
from os.path import join as pjoin
def download_pricing_file(file_url=DEFAULT_FILE_URL_S3_BUCKET, file_path=CUSTOM_PRICING_FILE_PATH):
    """
    Download pricing file from the file_url and save it to file_path.

    :type file_url: ``str``
    :param file_url: URL pointing to the pricing file.

    :type file_path: ``str``
    :param file_path: Path where a download pricing file will be saved.
    """
    from libcloud.utils.connection import get_response_object
    dir_name = os.path.dirname(file_path)
    if not os.path.exists(dir_name):
        msg = "Can't write to {}, directory {}, doesn't exist".format(file_path, dir_name)
        raise ValueError(msg)
    if os.path.exists(file_path) and os.path.isdir(file_path):
        msg = "Can't write to %s file path because it's a directory" % file_path
        raise ValueError(msg)
    response = get_response_object(file_url)
    body = response.body
    try:
        data = json.loads(body)
    except JSONDecodeError:
        msg = "Provided URL doesn't contain valid pricing data"
        raise Exception(msg)
    if not data.get('updated', None):
        msg = "Provided URL doesn't contain valid pricing data"
        raise Exception(msg)
    with open(file_path, 'w') as file_handle:
        file_handle.write(body)