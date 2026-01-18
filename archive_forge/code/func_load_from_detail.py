import contextlib
from oslo_utils import importutils
from oslo_utils import reflection
import stevedore.driver
from taskflow import exceptions as exc
from taskflow import logging
from taskflow.persistence import backends as p_backends
from taskflow.utils import misc
from taskflow.utils import persistence_utils as p_utils
def load_from_detail(flow_detail, store=None, backend=None, namespace=ENGINES_NAMESPACE, engine=ENGINE_DEFAULT, **kwargs):
    """Reloads an engine previously saved.

    This reloads the flow using the
    :func:`flow_from_detail() <flow_from_detail>` function and then calls
    into the :func:`load() <load>` function to create an engine from that flow.

    :param flow_detail: FlowDetail that holds state of the flow to load

    Further arguments are interpreted as for :func:`load() <load>`.

    :returns: engine
    """
    flow = flow_from_detail(flow_detail)
    return load(flow, flow_detail=flow_detail, store=store, backend=backend, namespace=namespace, engine=engine, **kwargs)