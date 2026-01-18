import logging
import time
import weakref
from botocore import xform_name
from botocore.exceptions import BotoCoreError, ConnectionError, HTTPClientError
from botocore.model import OperationNotFoundError
from botocore.utils import CachedProperty
def block_endpoint_discovery_required_operations(model, **kwargs):
    endpoint_discovery = model.endpoint_discovery
    if endpoint_discovery and endpoint_discovery.get('required'):
        raise EndpointDiscoveryRequired()