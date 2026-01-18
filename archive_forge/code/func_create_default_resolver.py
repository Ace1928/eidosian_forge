import copy
import logging
import os
import platform
import socket
import warnings
import botocore.client
import botocore.configloader
import botocore.credentials
import botocore.tokens
from botocore import (
from botocore.compat import HAS_CRT, MutableMapping
from botocore.configprovider import (
from botocore.errorfactory import ClientExceptionsFactory
from botocore.exceptions import (
from botocore.hooks import (
from botocore.loaders import create_loader
from botocore.model import ServiceModel
from botocore.parsers import ResponseParserFactory
from botocore.regions import EndpointResolver
from botocore.useragent import UserAgentString
from botocore.utils import (
from botocore.compat import HAS_CRT  # noqa
def create_default_resolver():
    loader = self.get_component('data_loader')
    endpoints, path = loader.load_data_with_path('endpoints')
    uses_builtin = loader.is_builtin_path(path)
    return EndpointResolver(endpoints, uses_builtin_data=uses_builtin)