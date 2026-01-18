import collections
import copy
import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from numbers import Number, Real
from typing import Any, Dict, List, Optional, Tuple, Union
import ray
import ray._private.services as services
from ray._private.utils import (
from ray.autoscaler._private import constants
from ray.autoscaler._private.cli_logger import cli_logger
from ray.autoscaler._private.docker import validate_docker_config
from ray.autoscaler._private.local.config import prepare_local
from ray.autoscaler._private.providers import _get_default_config
from ray.autoscaler.tags import NODE_TYPE_LEGACY_HEAD, NODE_TYPE_LEGACY_WORKER
def hash_runtime_conf(file_mounts, cluster_synced_files, extra_objs, generate_file_mounts_contents_hash=False):
    """Returns two hashes, a runtime hash and file_mounts_content hash.

    The runtime hash is used to determine if the configuration or file_mounts
    contents have changed. It is used at launch time (ray up) to determine if
    a restart is needed.

    The file_mounts_content hash is used to determine if the file_mounts or
    cluster_synced_files contents have changed. It is used at monitor time to
    determine if additional file syncing is needed.
    """
    runtime_hasher = hashlib.sha1()
    contents_hasher = hashlib.sha1()

    def add_content_hashes(path, allow_non_existing_paths: bool=False):

        def add_hash_of_file(fpath):
            with open(fpath, 'rb') as f:
                for chunk in iter(lambda: f.read(2 ** 20), b''):
                    contents_hasher.update(chunk)
        path = os.path.expanduser(path)
        if allow_non_existing_paths and (not os.path.exists(path)):
            return
        if os.path.isdir(path):
            dirs = []
            for dirpath, _, filenames in os.walk(path):
                dirs.append((dirpath, sorted(filenames)))
            for dirpath, filenames in sorted(dirs):
                contents_hasher.update(dirpath.encode('utf-8'))
                for name in filenames:
                    contents_hasher.update(name.encode('utf-8'))
                    fpath = os.path.join(dirpath, name)
                    add_hash_of_file(fpath)
        else:
            add_hash_of_file(path)
    conf_str = json.dumps(file_mounts, sort_keys=True).encode('utf-8') + json.dumps(extra_objs, sort_keys=True).encode('utf-8')
    if conf_str not in _hash_cache or generate_file_mounts_contents_hash:
        for local_path in sorted(file_mounts.values()):
            add_content_hashes(local_path)
        head_node_contents_hash = contents_hasher.hexdigest()
        if conf_str not in _hash_cache:
            runtime_hasher.update(conf_str)
            runtime_hasher.update(head_node_contents_hash.encode('utf-8'))
            _hash_cache[conf_str] = runtime_hasher.hexdigest()
        if cluster_synced_files is not None:
            for local_path in sorted(cluster_synced_files):
                add_content_hashes(local_path, allow_non_existing_paths=True)
        file_mounts_contents_hash = contents_hasher.hexdigest()
    else:
        file_mounts_contents_hash = None
    return (_hash_cache[conf_str], file_mounts_contents_hash)