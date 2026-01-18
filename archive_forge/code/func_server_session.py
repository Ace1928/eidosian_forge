from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote_plus, urlparse
from ..core.templates import AUTOLOAD_REQUEST_TAG, FILE
from ..resources import DEFAULT_SERVER_HTTP_URL
from ..util.serialization import make_globally_unique_css_safe_id
from ..util.strings import format_docstring
from .bundle import bundle_for_objs_and_resources
from .elements import html_page_for_render_items
from .util import RenderItem
def server_session(model: Model | None=None, session_id: ID | None=None, url: str='default', relative_urls: bool=False, resources: Literal['default'] | None='default', headers: dict[str, str]={}) -> str:
    """ Return a script tag that embeds content from a specific existing session on
    a Bokeh server.

    This function is typically only useful for serving from a a specific session
    that was previously created using the ``bokeh.client`` API.

    Bokeh apps embedded using these methods will NOT set the browser window title.

    .. note::
        Typically you will not want to save or re-use the output of this
        function for different or multiple page loads.

    Args:
        model (Model or None, optional) :
            The object to render from the session, or None. (default: None)

            If None, the entire document will be rendered.

        session_id (str) :
            A server session ID

        url (str, optional) :
            A URL to a Bokeh application on a Bokeh server (default: "default")

            If ``"default"`` the default URL ``{DEFAULT_SERVER_HTTP_URL}`` will be used.

        relative_urls (bool, optional) :
            Whether to use relative URLs for resources.

            If ``True`` the links generated for resources such a BokehJS
            JavaScript and CSS will be relative links.

            This should normally be set to ``False``, but must be set to
            ``True`` in situations where only relative URLs will work. E.g.
            when running the Bokeh behind reverse-proxies under certain
            configurations

        resources (str) : A string specifying what resources need to be loaded
            along with the document.

            If ``default`` then the default JS/CSS bokeh files will be loaded.

            If None then none of the resource files will be loaded. This is
            useful if you prefer to serve those resource files via other means
            (e.g. from a caching server). Be careful, however, that the resource
            files you'll load separately are of the same version as that of the
            server's, otherwise the rendering may not work correctly.

       headers (dict[str, str], optional) :
            A dictionary of key/values to be passed as HTTP Headers
            to Bokeh application code (default: None)

    Returns:
        A ``<script>`` tag that will embed content from a Bokeh Server.

        .. warning::
            It is typically a bad idea to re-use the same ``session_id`` for
            every page load. This is likely to create scalability and security
            problems, and will cause "shared Google doc" behavior, which is
            probably not desired.

    """
    if session_id is None:
        raise ValueError('Must supply a session_id')
    url = _clean_url(url)
    app_path = _get_app_path(url)
    elementid = make_globally_unique_css_safe_id()
    modelid = '' if model is None else model.id
    src_path = _src_path(url, elementid)
    src_path += _process_app_path(app_path)
    src_path += _process_relative_urls(relative_urls, url)
    src_path += _process_resources(resources)
    headers = dict(headers) if headers else {}
    headers['Bokeh-Session-Id'] = session_id
    tag = AUTOLOAD_REQUEST_TAG.render(src_path=src_path, app_path=app_path, elementid=elementid, modelid=modelid, headers=headers)
    return tag