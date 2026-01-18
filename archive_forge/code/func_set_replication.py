import logging
import os
import secrets
import shutil
import tempfile
import uuid
from contextlib import suppress
from urllib.parse import quote
import requests
from ..spec import AbstractBufferedFile, AbstractFileSystem
from ..utils import infer_storage_options, tokenize
def set_replication(self, path, replication):
    """
        Set file replication factor

        Parameters
        ----------
        path: str
            File location (not for directories)
        replication: int
            Number of copies of file on the cluster. Should be smaller than
            number of data nodes; normally 3 on most systems.
        """
    self._call('SETREPLICATION', path=path, method='put', replication=replication)