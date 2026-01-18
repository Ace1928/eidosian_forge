import logging
import time
import jmespath
from botocore.docs.docstring import WaiterDocstring
from botocore.utils import get_service_module_name
from . import xform_name
from .exceptions import ClientError, WaiterConfigError, WaiterError
def create_waiter_with_client(waiter_name, waiter_model, client):
    """

    :type waiter_name: str
    :param waiter_name: The name of the waiter.  The name should match
        the name (including the casing) of the key name in the waiter
        model file (typically this is CamelCasing).

    :type waiter_model: botocore.waiter.WaiterModel
    :param waiter_model: The model for the waiter configuration.

    :type client: botocore.client.BaseClient
    :param client: The botocore client associated with the service.

    :rtype: botocore.waiter.Waiter
    :return: The waiter object.

    """
    single_waiter_config = waiter_model.get_waiter(waiter_name)
    operation_name = xform_name(single_waiter_config.operation)
    operation_method = NormalizedOperationMethod(getattr(client, operation_name))

    def wait(self, **kwargs):
        Waiter.wait(self, **kwargs)
    wait.__doc__ = WaiterDocstring(waiter_name=waiter_name, event_emitter=client.meta.events, service_model=client.meta.service_model, service_waiter_model=waiter_model, include_signature=False)
    waiter_class_name = str('%s.Waiter.%s' % (get_service_module_name(client.meta.service_model), waiter_name))
    documented_waiter_cls = type(waiter_class_name, (Waiter,), {'wait': wait})
    return documented_waiter_cls(waiter_name, single_waiter_config, operation_method)