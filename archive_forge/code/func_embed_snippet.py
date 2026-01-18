import json
import re
from .widgets import Widget, DOMWidget, widget as widget_module
from .widgets.widget_link import Link
from .widgets.docutils import doc_subst
from ._version import __html_manager_version__
@doc_subst(_doc_snippets)
def embed_snippet(views, drop_defaults=True, state=None, indent=2, embed_url=None, requirejs=True, cors=True):
    """Return a snippet that can be embedded in an HTML file.

    Parameters
    ----------
    {views_attribute}
    {embed_kwargs}

    Returns
    -------
    A unicode string with an HTML snippet containing several `<script>` tags.
    """
    data = embed_data(views, drop_defaults=drop_defaults, state=state)
    widget_views = '\n'.join((widget_view_template.format(view_spec=escape_script(json.dumps(view_spec))) for view_spec in data['view_specs']))
    if embed_url is None:
        embed_url = DEFAULT_EMBED_REQUIREJS_URL if requirejs else DEFAULT_EMBED_SCRIPT_URL
    load = load_requirejs_template if requirejs else load_template
    use_cors = ' crossorigin="anonymous"' if cors else ''
    values = {'load': load.format(embed_url=embed_url, use_cors=use_cors), 'json_data': escape_script(json.dumps(data['manager_state'], indent=indent)), 'widget_views': widget_views}
    return snippet_template.format(**values)