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