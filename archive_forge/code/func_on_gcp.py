import enum
import os
import sys
import requests
from six.moves.urllib import request
from tensorflow.python.eager import context
from tensorflow.python.platform import tf_logging as logging
def on_gcp():
    """Detect whether the current running environment is on GCP."""
    gce_metadata_endpoint = 'http://' + os.environ.get(_GCE_METADATA_URL_ENV_VARIABLE, 'metadata.google.internal')
    try:
        response = requests.get('%s/computeMetadata/v1/%s' % (gce_metadata_endpoint, 'instance/hostname'), headers=GCP_METADATA_HEADER, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False