import os
import re
import urllib
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional
from ray._private.client_mode_hook import client_mode_hook
from ray._private.utils import _add_creatable_buckets_param_if_s3_uri, load_class
from ray._private.auto_init_hook import wrap_auto_init
@wrap_auto_init
@client_mode_hook
def get_filesystem() -> ('pyarrow.fs.FileSystem', str):
    """Initialize and get the configured storage filesystem, if possible.

    This method can be called from any Ray worker to get a reference to the configured
    storage filesystem.

    Examples:
        .. testcode::

            import ray
            from ray._private import storage

            ray.shutdown()

            ray.init(storage="/tmp/storage/cluster_1/storage")
            fs, path = storage.get_filesystem()
            print(fs)
            print(path)

        .. testoutput::

            <pyarrow._fs.LocalFileSystem object at ...>
            /tmp/storage/cluster_1/storage

    Returns:
        Tuple of pyarrow filesystem instance and the path under which files should
        be created for this cluster.

    Raises:
        RuntimeError if storage has not been configured or init failed.
    """
    return _get_filesystem_internal()