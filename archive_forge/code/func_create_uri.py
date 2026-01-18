from __future__ import annotations
import os
import json
import socket
import contextlib
import logging
from typing import Optional, Dict, Any, Union, Type, Mapping, Callable, List
from lazyops.utils.logs import default_logger as logger
import aiokeydb.v2.exceptions as exceptions
from aiokeydb.v2.types import BaseSettings, validator, root_validator, lazyproperty, KeyDBUri
from aiokeydb.v2.serializers import SerializerType, BaseSerializer
from aiokeydb.v2.utils import import_string
from aiokeydb.v2.configs.worker import KeyDBWorkerSettings
from aiokeydb.v2.backoff import default_backoff
def create_uri(self, name: str=None, uri: str=None, host: str=None, port: int=None, db_id: int=None, username: str=None, password: str=None, protocol: str=None, with_auth: bool=True) -> KeyDBUri:
    """
        Creates a URI from the given parameters
        """
    if not uri and (not host):
        uri = self.base_uri
    elif host:
        uri = f'{protocol or self.protocol}://{host}:{port or self.port}'
    if with_auth and '@' not in uri:
        if username and password:
            uri = f'{username}:{password}@{uri}'
        elif self.auth_str:
            uri = f'{self.auth_str}@{uri}'
    if '/' in uri[-4:]:
        split = uri.rsplit('/', 1)
        if db_id is None:
            db_id = int(split[1])
        uri = split[0]
    if db_id is None and name:
        db_id = self.db_mapping.get(name)
    db_id = db_id or self.db_id
    uri = f'{uri}/{db_id}'
    return KeyDBUri(dsn=uri)