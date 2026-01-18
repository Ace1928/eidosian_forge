from __future__ import annotations
import logging # isort:skip
from html import escape
from typing import TYPE_CHECKING, Any
from ..core.json_encoder import serialize_json
from ..core.templates import (
from ..document.document import DEFAULT_TITLE
from ..settings import settings
from ..util.serialization import make_globally_unique_css_safe_id
from .util import RenderItem
from .wrappers import wrap_in_onload, wrap_in_safely, wrap_in_script_tag
def html_page_for_render_items(bundle: Bundle | tuple[str, str], docs_json: dict[ID, DocJson], render_items: list[RenderItem], title: str | None, template: Template | str | None=None, template_variables: dict[str, Any]={}) -> str:
    """ Render an HTML page from a template and Bokeh render items.

    Args:
        bundle (tuple):
            A tuple containing (bokeh_js, bokeh_css).

        docs_json (JSON-like):
            Serialized Bokeh Document.

        render_items (RenderItems):
            Specific items to render from the document and where.

        title (str or None):
            A title for the HTML page. If None, DEFAULT_TITLE is used.

        template (str or Template or None, optional):
            A Template to be used for the HTML page. If None, FILE is used.

        template_variables (dict, optional):
            Any Additional variables to pass to the template.

    Returns:
        str

    """
    if title is None:
        title = DEFAULT_TITLE
    bokeh_js, bokeh_css = bundle
    json_id = make_globally_unique_css_safe_id()
    json = escape(serialize_json(docs_json), quote=False)
    json = wrap_in_script_tag(json, 'application/json', json_id)
    script = wrap_in_script_tag(script_for_render_items(json_id, render_items))
    context = template_variables.copy()
    context.update(dict(title=title, bokeh_js=bokeh_js, bokeh_css=bokeh_css, plot_script=json + script, docs=render_items, base=FILE, macros=MACROS))
    if len(render_items) == 1:
        context['doc'] = context['docs'][0]
        context['roots'] = context['doc'].roots
    context['plot_div'] = '\n'.join((div_for_render_item(item) for item in render_items))
    if template is None:
        template = FILE
    elif isinstance(template, str):
        template = get_env().from_string('{% extends base %}\n' + template)
    html = template.render(context)
    return html