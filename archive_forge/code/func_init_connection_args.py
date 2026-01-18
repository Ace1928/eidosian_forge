from __future__ import annotations
import asyncio
import json
import logging
import os
import typing as ty
from abc import ABC, ABCMeta, abstractmethod
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.cookies import SimpleCookie
from socket import gaierror
from jupyter_events import EventLogger
from tornado import web
from tornado.httpclient import AsyncHTTPClient, HTTPClientError, HTTPResponse
from traitlets import (
from traitlets.config import LoggingConfigurable, SingletonConfigurable
from jupyter_server import DEFAULT_EVENTS_SCHEMA_PATH, JUPYTER_SERVER_EVENTS_URI
def init_connection_args(self):
    """Initialize arguments used on every request.  Since these are primarily static values,
        we'll perform this operation once.
        """
    minimum_request_timeout = float(GatewayClient.KERNEL_LAUNCH_TIMEOUT) + self.launch_timeout_pad
    if self.request_timeout < minimum_request_timeout:
        self.request_timeout = minimum_request_timeout
    elif self.request_timeout > minimum_request_timeout:
        GatewayClient.KERNEL_LAUNCH_TIMEOUT = int(self.request_timeout - self.launch_timeout_pad)
    os.environ['KERNEL_LAUNCH_TIMEOUT'] = str(GatewayClient.KERNEL_LAUNCH_TIMEOUT)
    if self.headers:
        self._connection_args['headers'] = json.loads(self.headers)
        if self.auth_header_key not in self._connection_args['headers']:
            self._connection_args['headers'].update({f'{self.auth_header_key}': f'{self.auth_scheme} {self.auth_token}'})
    self._connection_args['connect_timeout'] = self.connect_timeout
    self._connection_args['request_timeout'] = self.request_timeout
    self._connection_args['validate_cert'] = self.validate_cert
    if self.client_cert:
        self._connection_args['client_cert'] = self.client_cert
        self._connection_args['client_key'] = self.client_key
        if self.ca_certs:
            self._connection_args['ca_certs'] = self.ca_certs
    if self.http_user:
        self._connection_args['auth_username'] = self.http_user
    if self.http_pwd:
        self._connection_args['auth_password'] = self.http_pwd