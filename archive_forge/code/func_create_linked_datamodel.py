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
def create_linked_datamodel(obj, root=None):
    """
    Creates a Bokeh DataModel from a Parameterized class or instance
    which automatically links the parameters bi-directionally.

    Arguments
    ---------
    obj: param.Parameterized
       The Parameterized class to create a linked DataModel for.

    Returns
    -------
    DataModel instance linked to the Parameterized object.
    """
    if isinstance(obj, type) and issubclass(obj, pm.Parameterized):
        cls = obj
    elif isinstance(obj, pm.Parameterized):
        cls = type(obj)
    else:
        raise TypeError('Can only create DataModel for Parameterized class or instance.')
    if cls in _DATA_MODELS:
        model = _DATA_MODELS[cls]
    else:
        _DATA_MODELS[cls] = model = construct_data_model(obj)
    properties = model.properties()
    model = model(**{k: v for k, v in obj.param.values().items() if k in properties})
    _changing = []

    def cb_bokeh(attr, old, new):
        if attr in _changing:
            return
        try:
            _changing.append(attr)
            obj.param.update(**{attr: new})
        finally:
            _changing.remove(attr)

    def cb_param(*events):
        update = {event.name: event.new for event in events if event.name not in _changing}
        try:
            _changing.extend(list(update))
            tags = [tag for tag in model.tags if tag.startswith('__ref:')]
            if root:
                ref = root.ref['id']
            elif tags:
                ref = tags[0].split('__ref:')[-1]
            else:
                ref = None
            if ref and ref in state._views:
                _, root_model, doc, comm = state._views[ref]
                if comm or state._unblocked(doc):
                    with unlocked():
                        model.update(**update)
                    if comm and 'embedded' not in root_model.tags:
                        push(doc, comm)
                else:
                    cb = partial(model.update, **update)
                    if doc.session_context:
                        doc.add_next_tick_callback(cb)
                    else:
                        cb()
            else:
                model.update(**update)
        finally:
            for attr in update:
                _changing.remove(attr)
    for p in obj.param:
        if p in properties:
            model.on_change(p, cb_bokeh)
    obj.param.watch(cb_param, list(set(properties) & set(obj.param)))
    return model