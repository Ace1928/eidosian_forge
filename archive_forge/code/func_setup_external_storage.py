import abc
import logging
import os
import random
import shutil
import time
import urllib
import uuid
from collections import namedtuple
from typing import IO, List, Optional, Tuple
import ray
from ray._private.ray_constants import DEFAULT_OBJECT_PREFIX
from ray._raylet import ObjectRef
def setup_external_storage(config, session_name):
    """Setup the external storage according to the config."""
    global _external_storage
    if config:
        storage_type = config['type']
        if storage_type == 'filesystem':
            _external_storage = FileSystemStorage(**config['params'])
        elif storage_type == 'ray_storage':
            _external_storage = ExternalStorageRayStorageImpl(session_name, **config['params'])
        elif storage_type == 'smart_open':
            _external_storage = ExternalStorageSmartOpenImpl(**config['params'])
        elif storage_type == 'mock_distributed_fs':
            _external_storage = FileSystemStorage(**config['params'])
        elif storage_type == 'unstable_fs':
            _external_storage = UnstableFileStorage(**config['params'])
        elif storage_type == 'slow_fs':
            _external_storage = SlowFileStorage(**config['params'])
        else:
            raise ValueError(f'Unknown external storage type: {storage_type}')
    else:
        _external_storage = NullStorage()
    return _external_storage