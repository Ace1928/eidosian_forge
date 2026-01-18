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
def json_item(model: Model, target: ID | None=None, theme: ThemeLike=None) -> StandaloneEmbedJson:
    """ Return a JSON block that can be used to embed standalone Bokeh content.

    Args:
        model (Model) :
            The Bokeh object to embed

        target (string, optional)
            A div id to embed the model into. If None, the target id must
            be supplied in the JavaScript call.

        theme (Theme, optional) :
            Applies the specified theme to the created html. If ``None``, or
            not specified, and the function is passed a document or the full set
            of roots of a document, applies the theme of that document.  Otherwise
            applies the default theme.

    Returns:
        JSON-like

    This function returns a JSON block that can be consumed by the BokehJS
    function ``Bokeh.embed.embed_item``. As an example, a Flask endpoint for
    ``/plot`` might return the following content to embed a Bokeh plot into
    a div with id *"myplot"*:

    .. code-block:: python

        @app.route('/plot')
        def plot():
            p = make_plot('petal_width', 'petal_length')
            return json.dumps(json_item(p, "myplot"))

    Then a web page can retrieve this JSON and embed the plot by calling
    ``Bokeh.embed.embed_item``:

    .. code-block:: html

        <script>
        fetch('/plot')
            .then(function(response) { return response.json(); })
            .then(function(item) { Bokeh.embed.embed_item(item); })
        </script>

    Alternatively, if is more convenient to supply the target div id directly
    in the page source, that is also possible. If `target_id` is omitted in the
    call to this function:

    .. code-block:: python

        return json.dumps(json_item(p))

    Then the value passed to ``embed_item`` is used:

    .. code-block:: javascript

        Bokeh.embed.embed_item(item, "myplot");

    """
    with OutputDocumentFor([model], apply_theme=theme) as doc:
        doc.title = ''
        [doc_json] = standalone_docs_json([model]).values()
    root_id = doc_json['roots'][0]['id']
    return StandaloneEmbedJson(target_id=target, root_id=root_id, doc=doc_json, version=__version__)