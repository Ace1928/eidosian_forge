import socket
import re
import logging
import warnings
from requests.exceptions import RequestException, SSLError
import http.client as http_client
from urllib.parse import quote, unquote
from urllib.parse import urljoin, urlparse, urlunparse
from time import sleep, time
from swiftclient import version as swiftclient_version
from swiftclient.exceptions import ClientException
from swiftclient.requests_compat import SwiftClientRequestsSession
from swiftclient.utils import (
def get_service_auth(self):
    opts = self.os_options
    service_options = {}
    service_options['tenant_name'] = opts.get('service_project_name', None)
    service_options['region_name'] = opts.get('region_name', None)
    service_options['object_storage_url'] = opts.get('object_storage_url', None)
    service_user = opts.get('service_username', None)
    service_key = opts.get('service_key', None)
    return get_auth(self.authurl, service_user, service_key, session=self.session, snet=self.snet, auth_version=self.auth_version, os_options=service_options, cacert=self.cacert, insecure=self.insecure, timeout=self.timeout)