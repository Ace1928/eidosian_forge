import os
import posixpath
import sys
import threading
import urllib.parse
from contextlib import contextmanager
from mlflow.entities import FileInfo
from mlflow.store.artifact.artifact_repo import ArtifactRepository
@contextmanager
def get_sfp_connection(self):
    with self._cond:
        self._cond.wait_for(lambda: bool(self._connections))
        connection = self._connections.pop(-1)
    yield connection
    self._connections.append(connection)