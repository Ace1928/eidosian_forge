import abc
import contextlib
import copy
import hashlib
import os
import threading
from oslo_utils import reflection
from oslo_utils import strutils
import requests
from novaclient import exceptions
from novaclient import utils
@contextlib.contextmanager
def alternate_service_type(self, default, allowed_types=()):
    original_service_type = self.api.client.service_type
    if original_service_type in allowed_types:
        yield
    else:
        self.api.client.service_type = default
        try:
            yield
        finally:
            self.api.client.service_type = original_service_type