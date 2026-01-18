import json
import sys
from unittest import mock
from keystoneauth1 import fixture
import requests
def is_volume_endpoint_enabled(self, client):
    return self.volume_endpoint_enabled