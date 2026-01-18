import copy
import logging
from botocore import xform_name
from botocore.utils import merge_dicts
from .action import BatchAction
from .params import create_request_parameters
from .response import ResourceHandler
from ..docs import docstring
def load_from_definition(self, resource_name, collection_model, service_context, event_emitter):
    """
        Loads a collection from a model, creating a new
        :py:class:`CollectionManager` subclass
        with the correct properties and methods, named based on the service
        and resource name, e.g. ec2.InstanceCollectionManager. It also
        creates a new :py:class:`ResourceCollection` subclass which is used
        by the new manager class.

        :type resource_name: string
        :param resource_name: Name of the resource to look up. For services,
                              this should match the ``service_name``.

        :type service_context: :py:class:`~boto3.utils.ServiceContext`
        :param service_context: Context about the AWS service

        :type event_emitter: :py:class:`~botocore.hooks.HierarchialEmitter`
        :param event_emitter: An event emitter

        :rtype: Subclass of :py:class:`CollectionManager`
        :return: The collection class.
        """
    attrs = {}
    collection_name = collection_model.name
    self._load_batch_actions(attrs, resource_name, collection_model, service_context.service_model, event_emitter)
    self._load_documented_collection_methods(attrs=attrs, resource_name=resource_name, collection_model=collection_model, service_model=service_context.service_model, event_emitter=event_emitter, base_class=ResourceCollection)
    if service_context.service_name == resource_name:
        cls_name = '{0}.{1}Collection'.format(service_context.service_name, collection_name)
    else:
        cls_name = '{0}.{1}.{2}Collection'.format(service_context.service_name, resource_name, collection_name)
    collection_cls = type(str(cls_name), (ResourceCollection,), attrs)
    self._load_documented_collection_methods(attrs=attrs, resource_name=resource_name, collection_model=collection_model, service_model=service_context.service_model, event_emitter=event_emitter, base_class=CollectionManager)
    attrs['_collection_cls'] = collection_cls
    cls_name += 'Manager'
    return type(str(cls_name), (CollectionManager,), attrs)