import copy
import sys
from openstack.config import loader
from openstack.config.loader import *  # noqa
from os_client_config import cloud_config
from os_client_config import defaults
def get_cache_max_age(self):
    return self.get_cache_expiration_time()