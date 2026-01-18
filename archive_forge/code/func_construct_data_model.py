import weakref
from functools import partial
import bokeh
import bokeh.core.properties as bp
import param as pm
from bokeh.model import DataModel
from bokeh.models import ColumnDataSource
from ..reactive import Syncable
from .document import unlocked
from .notebook import push
from .state import state
def construct_data_model(parameterized, name=None, ignore=[], types={}):
    """
    Dynamically creates a Bokeh DataModel class from a Parameterized
    object.

    Arguments
    ---------
    parameterized: param.Parameterized
        The Parameterized class or instance from which to create the
        DataModel
    name: str or None
        Name of the dynamically created DataModel class
    ignore: list(str)
        List of parameters to ignore.
    types: dict
        A dictionary mapping from parameter name to a Parameter type,
        making it possible to override the default parameter types.

    Returns
    -------
    DataModel
    """
    properties = {}
    for pname in parameterized.param:
        if pname in ignore:
            continue
        p = parameterized.param[pname]
        if p.precedence and p.precedence < 0:
            continue
        ptype = types.get(pname, type(p))
        prop = PARAM_MAPPING.get(ptype)
        if isinstance(parameterized, Syncable):
            pname = parameterized._rename.get(pname, pname)
        if pname == 'name' or pname is None:
            continue
        nullable = getattr(p, 'allow_None', False)
        kwargs = {'default': p.default, 'help': p.doc}
        if prop is None:
            bk_prop, accepts = (bp.Any(**kwargs), [])
        else:
            bkp = prop(p, {} if nullable else kwargs)
            bk_prop, accepts = bkp if isinstance(bkp, tuple) else (bkp, [])
            if nullable:
                bk_prop = bp.Nullable(bk_prop, **kwargs)
        for bkp, convert in accepts:
            bk_prop = bk_prop.accepts(bkp, convert)
        properties[pname] = bk_prop
    name = name or parameterized.name
    return type(name, (DataModel,), properties)