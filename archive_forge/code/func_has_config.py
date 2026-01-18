from datetime import datetime, timedelta
from tests.compat import mock, unittest
import os
from boto import provider
from boto.compat import expanduser
from boto.exception import InvalidInstanceMetadataError
def has_config(self, section_name, key):
    try:
        self.config[section_name][key]
        return True
    except KeyError:
        return False