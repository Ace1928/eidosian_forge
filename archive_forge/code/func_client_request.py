import logging
import time
import requests
from oslo_utils import importutils
from troveclient.apiclient import exceptions
def client_request(self, method, url, **kwargs):
    return self.http_client.client_request(self, method, url, **kwargs)