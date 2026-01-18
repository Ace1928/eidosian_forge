import os
from traitlets import Any, Unicode, List, Dict, Union, Instance
from ipywidgets import DOMWidget
from ipywidgets.widgets.widget import widget_serialization
from .Template import Template, get_template
from ._version import semver
from .ForceLoad import force_load_instance
import inspect
from importlib import import_module
def create_ref_and_observe(traitlet):
    data = traitlet.get(self)
    ref_name = traitlet.name + '_ref'
    self.add_traits(**{ref_name: Any(as_refs(traitlet.name, data)).tag(sync=True)})

    def on_ref_source_change(change):
        setattr(self, ref_name, as_refs(traitlet.name, change['new']))
    self.observe(on_ref_source_change, traitlet.name)