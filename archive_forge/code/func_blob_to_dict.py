from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import os
def blob_to_dict(blob):
    return {'bucket': {'name': blob.bucket.path}, 'cache_control': blob.cache_control, 'chunk_size': blob.chunk_size, 'media_link': blob.media_link, 'self_link': blob.self_link, 'storage_class': blob.storage_class}