from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Union
from ..core.json_encoder import serialize_json
from ..core.templates import DOC_NB_JS
from ..document import Document
from ..model import Model
from ..themes import Theme
from .elements import div_for_render_item
from .util import FromCurdoc, OutputDocumentFor, standalone_docs_json_and_render_items
def notebook_content(model: Model, notebook_comms_target: str | None=None, theme: ThemeSource=FromCurdoc) -> tuple[str, str, Document]:
    """ Return script and div that will display a Bokeh plot in a Jupyter
    Notebook.

    The data for the plot is stored directly in the returned HTML.

    Args:
        model (Model) : Bokeh object to render

        notebook_comms_target (str, optional) :
            A target name for a Jupyter Comms object that can update
            the document that is rendered to this notebook div

        theme (Theme, optional) :
            Defaults to the ``Theme`` instance in the current document.
            Setting this to ``None`` uses the default theme or the theme
            already specified in the document. Any other value must be an
            instance of the ``Theme`` class.

    Returns:
        script, div, Document

    .. note::
        Assumes :func:`~bokeh.io.notebook.load_notebook` or the equivalent
        has already been executed.

    """
    if not isinstance(model, Model):
        raise ValueError('notebook_content expects a single Model instance')
    with OutputDocumentFor([model], apply_theme=theme, always_new=True) as new_doc:
        docs_json, [render_item] = standalone_docs_json_and_render_items([model])
    div = div_for_render_item(render_item)
    render_item = render_item.to_json()
    if notebook_comms_target:
        render_item['notebook_comms_target'] = notebook_comms_target
    script = DOC_NB_JS.render(docs_json=serialize_json(docs_json), render_items=serialize_json([render_item]))
    return (script, div, new_doc)