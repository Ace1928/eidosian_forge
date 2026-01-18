import contextlib
from oslo_utils import importutils
from oslo_utils import reflection
import stevedore.driver
from taskflow import exceptions as exc
from taskflow import logging
from taskflow.persistence import backends as p_backends
from taskflow.utils import misc
from taskflow.utils import persistence_utils as p_utils
def save_factory_details(flow_detail, flow_factory, factory_args, factory_kwargs, backend=None):
    """Saves the given factories reimportable attributes into the flow detail.

    This function saves the factory name, arguments, and keyword arguments
    into the given flow details object  and if a backend is provided it will
    also ensure that the backend saves the flow details after being updated.

    :param flow_detail: FlowDetail that holds state of the flow to load
    :param flow_factory: function or string: function that creates the flow
    :param factory_args: list or tuple of factory positional arguments
    :param factory_kwargs: dict of factory keyword arguments
    :param backend: storage backend to use or configuration
    """
    if not factory_args:
        factory_args = []
    if not factory_kwargs:
        factory_kwargs = {}
    factory_name, _factory_fun = _fetch_validate_factory(flow_factory)
    factory_data = {'factory': {'name': factory_name, 'args': factory_args, 'kwargs': factory_kwargs}}
    if not flow_detail.meta:
        flow_detail.meta = factory_data
    else:
        flow_detail.meta.update(factory_data)
    if backend is not None:
        if isinstance(backend, dict):
            backend = p_backends.fetch(backend)
        with contextlib.closing(backend.get_connection()) as conn:
            conn.update_flow_details(flow_detail)