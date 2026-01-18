from __future__ import (absolute_import, division, print_function)
import json
import os
import tarfile
from ansible.module_utils.common.text.converters import to_native
def api_image_id(archive_image_id):
    """
    Accepts an image hash in the format stored in manifest.json, and returns an equivalent identifier
    that represents the same image hash, but in the format presented by the Docker Engine API.

    :param archive_image_id: plain image hash
    :type archive_image_id: str

    :returns: Prefixed hash used by REST api
    :rtype: str
    """
    return 'sha256:%s' % archive_image_id