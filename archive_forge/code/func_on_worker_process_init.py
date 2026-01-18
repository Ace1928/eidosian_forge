import os
import sys
import warnings
from datetime import datetime, timezone
from importlib import import_module
from typing import IO, TYPE_CHECKING, Any, List, Optional, cast
from kombu.utils.imports import symbol_by_name
from kombu.utils.objects import cached_property
from celery import _state, signals
from celery.exceptions import FixupWarning, ImproperlyConfigured
def on_worker_process_init(self, **kwargs: Any) -> None:
    if os.environ.get('FORKED_BY_MULTIPROCESSING'):
        self.validate_models()
    for c in self._db.connections.all():
        if c and c.connection:
            self._maybe_close_db_fd(c.connection)
    self._close_database(force=True)
    self.close_cache()