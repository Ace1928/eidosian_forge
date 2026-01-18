from SQS when you have short-running tasks (or a large number of workers).
from __future__ import annotations
import base64
import socket
import string
import uuid
from datetime import datetime
from queue import Empty
from botocore.client import Config
from botocore.exceptions import ClientError
from vine import ensure_promise, promise, transform
from kombu.asynchronous import get_event_loop
from kombu.asynchronous.aws.ext import boto3, exceptions
from kombu.asynchronous.aws.sqs.connection import AsyncSQSConnection
from kombu.asynchronous.aws.sqs.message import AsyncMessage
from kombu.log import get_logger
from kombu.utils import scheduling
from kombu.utils.encoding import bytes_to_str, safe_str
from kombu.utils.json import dumps, loads
from kombu.utils.objects import cached_property
from . import virtual
def new_sqs_client(self, region, access_key_id, secret_access_key, session_token=None):
    session = boto3.session.Session(region_name=region, aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key, aws_session_token=session_token)
    is_secure = self.is_secure if self.is_secure is not None else True
    client_kwargs = {'use_ssl': is_secure}
    if self.endpoint_url is not None:
        client_kwargs['endpoint_url'] = self.endpoint_url
    client_config = self.transport_options.get('client-config') or {}
    config = Config(**client_config)
    return session.client('sqs', config=config, **client_kwargs)