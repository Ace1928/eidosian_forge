from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
def convert_map_to_clusters(clusters):
    carray = []
    for key, cluster in clusters.items():
        cluster['name'] = key.split('/')[-1]
        carray.append(cluster)
    return carray