import logging
from functools import partial
from .action import ServiceAction
from .action import WaiterAction
from .base import ResourceMeta, ServiceResource
from .collection import CollectionFactory
from .model import ResourceModel
from .response import build_identifiers, ResourceHandler
from ..exceptions import ResourceLoadException
from ..docs import docstring
def do_action(self, *args, **kwargs):
    response = action(self, *args, **kwargs)
    if hasattr(self, 'load'):
        self.meta.data = None
    return response