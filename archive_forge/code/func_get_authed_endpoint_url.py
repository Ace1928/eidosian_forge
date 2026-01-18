import json
from unittest import mock
import uuid
import requests
from cinderclient import client
from cinderclient import exceptions
from cinderclient.tests.unit import utils
def get_authed_endpoint_url(retries=0):
    cl = client.HTTPClient('username', 'password', 'project_id', 'auth_test', os_endpoint='volume/v100/', retries=retries)
    cl.auth_token = 'token'
    return cl