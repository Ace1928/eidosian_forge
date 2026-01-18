from __future__ import (absolute_import, division, print_function)
import logging
import logging.config
import os
import tempfile
from datetime import datetime  # noqa: F401, pylint: disable=unused-import
from operator import eq
import time
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import iteritems
def bucket_details_factory(bucket_details_type, module):
    bucket_details = None
    if bucket_details_type == 'create':
        bucket_details = CreateBucketDetails()
    elif bucket_details_type == 'update':
        bucket_details = UpdateBucketDetails()
    bucket_details.compartment_id = module.params['compartment_id']
    bucket_details.name = module.params['name']
    bucket_details.public_access_type = module.params['public_access_type']
    bucket_details.metadata = module.params['metadata']
    return bucket_details