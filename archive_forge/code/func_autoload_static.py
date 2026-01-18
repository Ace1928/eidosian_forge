from __future__ import annotations
import logging # isort:skip
from typing import (
from .. import __version__
from ..core.templates import (
from ..document.document import DEFAULT_TITLE, Document
from ..model import Model
from ..resources import Resources, ResourcesLike
from ..themes import Theme
from .bundle import Script, bundle_for_objs_and_resources
from .elements import html_page_for_render_items, script_for_render_items
from .util import (
from .wrappers import wrap_in_onload
def autoload_static(model: Model | Document, resources: Resources, script_path: str) -> tuple[str, str]:
    """ Return JavaScript code and a script tag that can be used to embed
    Bokeh Plots.

    The data for the plot is stored directly in the returned JavaScript code.

    Args:
        model (Model or Document) :

        resources (Resources) :

        script_path (str) :

    Returns:
        (js, tag) :
            JavaScript code to be saved at ``script_path`` and a ``<script>``
            tag to load it

    Raises:
        ValueError

    """
    if isinstance(model, Model):
        models = [model]
    elif isinstance(model, Document):
        models = model.roots
    else:
        raise ValueError('autoload_static expects a single Model or Document')
    with OutputDocumentFor(models):
        docs_json, [render_item] = standalone_docs_json_and_render_items([model])
    bundle = bundle_for_objs_and_resources(None, resources)
    bundle.add(Script(script_for_render_items(docs_json, [render_item])))
    _, elementid = next(iter(render_item.roots.to_json().items()))
    js = wrap_in_onload(AUTOLOAD_JS.render(bundle=bundle, elementid=elementid))
    tag = AUTOLOAD_TAG.render(src_path=script_path, elementid=elementid)
    return (js, tag)