import logging
import time
import weakref
from botocore import xform_name
from botocore.exceptions import BotoCoreError, ConnectionError, HTTPClientError
from botocore.model import OperationNotFoundError
from botocore.utils import CachedProperty
def discover_endpoint(self, request, operation_name, **kwargs):
    ids = request.context.get('discovery', {}).get('identifiers')
    if ids is None:
        return
    endpoint = self._manager.describe_endpoint(Operation=operation_name, Identifiers=ids)
    if endpoint is None:
        logger.debug('Failed to discover and inject endpoint')
        return
    if not endpoint.startswith('http'):
        endpoint = 'https://' + endpoint
    logger.debug('Injecting discovered endpoint: %s', endpoint)
    request.url = endpoint