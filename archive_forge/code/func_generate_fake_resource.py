import inspect
import random
from typing import (
from unittest import mock
import uuid
from openstack import format as _format
from openstack import proxy
from openstack import resource
from openstack import service_description
def generate_fake_resource(resource_type: Type[Resource], **attrs: Dict[str, Any]) -> Resource:
    """Generate a fake resource

    :param type resource_type: Object class
    :param dict attrs: Optional attributes to be set on resource

    Example usage:

    .. code-block:: python

        >>> from openstack.compute.v2 import server
        >>> from openstack.test import fakes
        >>> fakes.generate_fake_resource(server.Server)
        openstack.compute.v2.server.Server(...)

    :param type resource_type: Object class
    :param dict attrs: Optional attributes to be set on resource
    :return: Instance of ``resource_type`` class populated with fake
        values of expected types
    :raises NotImplementedError: If a resource attribute specifies a ``type``
        or ``list_type`` that cannot be automatically generated
    """
    base_attrs: Dict[str, Any] = {}
    for name, value in inspect.getmembers(resource_type, predicate=lambda x: isinstance(x, (resource.Body, resource.URI))):
        if isinstance(value, resource.Body):
            target_type = value.type
            if target_type is None:
                if name == 'properties' and hasattr(resource_type, '_store_unknown_attrs_as_properties') and resource_type._store_unknown_attrs_as_properties:
                    base_attrs[name] = dict()
                else:
                    base_attrs[name] = uuid.uuid4().hex
            elif issubclass(target_type, resource.Resource):
                base_attrs[name] = generate_fake_resource(target_type)
            elif issubclass(target_type, list) and value.list_type is not None:
                item_type = value.list_type
                if issubclass(item_type, resource.Resource):
                    base_attrs[name] = [generate_fake_resource(item_type)]
                elif issubclass(item_type, dict):
                    base_attrs[name] = [{}]
                elif issubclass(item_type, str):
                    base_attrs[name] = [uuid.uuid4().hex]
                else:
                    msg = 'Fake value for %s.%s can not be generated' % (resource_type.__name__, name)
                    raise NotImplementedError(msg)
            elif issubclass(target_type, list) and value.list_type is None:
                base_attrs[name] = [uuid.uuid4().hex]
            elif issubclass(target_type, str):
                base_attrs[name] = uuid.uuid4().hex
            elif issubclass(target_type, int):
                base_attrs[name] = random.randint(1, 100)
            elif issubclass(target_type, float):
                base_attrs[name] = random.random()
            elif issubclass(target_type, bool) or issubclass(target_type, _format.BoolStr):
                base_attrs[name] = random.choice([True, False])
            elif issubclass(target_type, dict):
                base_attrs[name] = dict()
            else:
                msg = 'Fake value for %s.%s can not be generated' % (resource_type.__name__, name)
                raise NotImplementedError(msg)
        if isinstance(value, resource.URI):
            base_attrs[name] = uuid.uuid4().hex
    base_attrs.update(**attrs)
    fake = resource_type(**base_attrs)
    return fake