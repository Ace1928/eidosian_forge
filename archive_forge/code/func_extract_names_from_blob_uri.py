from __future__ import absolute_import, division, print_function
import base64
import random
import re
import time
from ansible.module_utils.basic import to_native, to_bytes
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import (AzureRMModuleBase,
def extract_names_from_blob_uri(blob_uri, storage_suffix):
    m = re.match('^https://(?P<accountname>[^.]+)\\.blob\\.{0}/(?P<containername>[^/]+)/(?P<blobname>.+)$'.format(storage_suffix), blob_uri)
    if not m:
        raise Exception("unable to parse blob uri '%s'" % blob_uri)
    extracted_names = m.groupdict()
    return extracted_names