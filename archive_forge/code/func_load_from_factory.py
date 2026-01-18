import contextlib
from oslo_utils import importutils
from oslo_utils import reflection
import stevedore.driver
from taskflow import exceptions as exc
from taskflow import logging
from taskflow.persistence import backends as p_backends
from taskflow.utils import misc
from taskflow.utils import persistence_utils as p_utils
def load_from_factory(flow_factory, factory_args=None, factory_kwargs=None, store=None, book=None, backend=None, namespace=ENGINES_NAMESPACE, engine=ENGINE_DEFAULT, **kwargs):
    """Loads a flow from a factory function into an engine.

    Gets flow factory function (or name of it) and creates flow with
    it. Then, the flow is loaded into an engine with the :func:`load() <load>`
    function, and the factory function fully qualified name is saved to flow
    metadata so that it can be later resumed.

    :param flow_factory: function or string: function that creates the flow
    :param factory_args: list or tuple of factory positional arguments
    :param factory_kwargs: dict of factory keyword arguments

    Further arguments are interpreted as for :func:`load() <load>`.

    :returns: engine
    """
    _factory_name, factory_fun = _fetch_validate_factory(flow_factory)
    if not factory_args:
        factory_args = []
    if not factory_kwargs:
        factory_kwargs = {}
    flow = factory_fun(*factory_args, **factory_kwargs)
    if isinstance(backend, dict):
        backend = p_backends.fetch(backend)
    flow_detail = p_utils.create_flow_detail(flow, book=book, backend=backend)
    save_factory_details(flow_detail, flow_factory, factory_args, factory_kwargs, backend=backend)
    return load(flow=flow, store=store, flow_detail=flow_detail, book=book, backend=backend, namespace=namespace, engine=engine, **kwargs)