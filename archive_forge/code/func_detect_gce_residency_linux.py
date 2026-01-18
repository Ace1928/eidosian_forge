import datetime
import http.client as http_client
import json
import logging
import os
from urllib.parse import urljoin
from google.auth import _helpers
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import metrics
def detect_gce_residency_linux():
    """Detect Google Compute Engine residency by smbios check on Linux

    Returns:
        bool: True if the GCE product name file is detected, False otherwise.
    """
    try:
        with open(_GCE_PRODUCT_NAME_FILE, 'r') as file_obj:
            content = file_obj.read().strip()
    except Exception:
        return False
    return content.startswith(_GOOGLE)