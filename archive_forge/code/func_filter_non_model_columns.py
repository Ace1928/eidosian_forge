import functools
from oslo_db import exception as db_exc
from oslo_utils import excutils
import sqlalchemy
from sqlalchemy.ext import associationproxy
from sqlalchemy.orm import exc
from sqlalchemy.orm import properties
from neutron_lib._i18n import _
from neutron_lib.api import attributes
from neutron_lib import exceptions as n_exc
def filter_non_model_columns(data, model):
    """Return the attributes from data which are model columns.

    :param data: The dict containing the data to filter.
    :param model: The model who's column names are used when filtering data.
    :returns: A new dict who's keys are columns in model or are association
        proxies of the model.
    """
    mapper = sqlalchemy.inspect(model)
    columns = set((c.name for c in mapper.columns))
    try:
        _association_proxy = associationproxy.ASSOCIATION_PROXY
    except AttributeError:
        _association_proxy = associationproxy.AssociationProxyExtensionType.ASSOCIATION_PROXY
    columns.update((d.value_attr for d in mapper.all_orm_descriptors if d.extension_type is _association_proxy))
    return dict(((k, v) for k, v in data.items() if k in columns))