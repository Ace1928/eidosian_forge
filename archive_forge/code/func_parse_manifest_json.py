import base64
import copy
import json
import os
import re
import shutil
import sys
import tempfile
import warnings
import zipfile
from io import BytesIO
from xml.dom import minidom
from typing_extensions import deprecated
from selenium.common.exceptions import WebDriverException
def parse_manifest_json(content):
    """Extracts the details from the contents of a WebExtensions
            `manifest.json` file."""
    manifest = json.loads(content)
    try:
        id = manifest['applications']['gecko']['id']
    except KeyError:
        id = manifest['name'].replace(' ', '') + '@' + manifest['version']
    return {'id': id, 'version': manifest['version'], 'name': manifest['version'], 'unpack': False}