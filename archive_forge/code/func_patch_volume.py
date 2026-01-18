from __future__ import (absolute_import, division, print_function)
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def patch_volume(rest_api, uuid, body, query=None, job_timeout=120):
    api = 'storage/volumes'
    return rest_generic.patch_async(rest_api, api, uuid, body, query=query, job_timeout=job_timeout)