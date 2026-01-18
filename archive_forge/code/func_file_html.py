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
def file_html(models: Model | Document | Sequence[Model], resources: ResourcesLike | None=None, title: str | None=None, *, template: Template | str=FILE, template_variables: dict[str, Any]={}, theme: ThemeLike=None, suppress_callback_warning: bool=False, _always_new: bool=False) -> str:
    """ Return an HTML document that embeds Bokeh Model or Document objects.

    The data for the plot is stored directly in the returned HTML, with
    support for customizing the JS/CSS resources independently and
    customizing the jinja2 template.

    Args:
        models (Model or Document or seq[Model]) : Bokeh object or objects to render
            typically a Model or Document

        resources (ResourcesLike) :
            A resources configuration for Bokeh JS & CSS assets.

        title (str, optional) :
            A title for the HTML document ``<title>`` tags or None. (default: None)

            If None, attempt to automatically find the Document title from the given
            plot objects.

        template (Template, optional) : HTML document template (default: FILE)
            A Jinja2 Template, see bokeh.core.templates.FILE for the required
            template parameters

        template_variables (dict, optional) : variables to be used in the Jinja2
            template. If used, the following variable names will be overwritten:
            title, bokeh_js, bokeh_css, plot_script, plot_div

        theme (Theme, optional) :
            Applies the specified theme to the created html. If ``None``, or
            not specified, and the function is passed a document or the full set
            of roots of a document, applies the theme of that document.  Otherwise
            applies the default theme.

        suppress_callback_warning (bool, optional) :
            Normally generating standalone HTML from a Bokeh Document that has
            Python callbacks will result in a warning stating that the callbacks
            cannot function. However, this warning can be suppressed by setting
            this value to True (default: False)

    Returns:
        UTF-8 encoded HTML

    """
    models_seq: Sequence[Model] = []
    if isinstance(models, Model):
        models_seq = [models]
    elif isinstance(models, Document):
        if len(models.roots) == 0:
            raise ValueError('Document has no root Models')
        models_seq = models.roots
    else:
        models_seq = models
    resources = Resources.build(resources)
    with OutputDocumentFor(models_seq, apply_theme=theme, always_new=_always_new) as doc:
        docs_json, render_items = standalone_docs_json_and_render_items(models_seq, suppress_callback_warning=suppress_callback_warning)
        title = _title_from_models(models_seq, title)
        bundle = bundle_for_objs_and_resources([doc], resources)
        return html_page_for_render_items(bundle, docs_json, render_items, title=title, template=template, template_variables=template_variables)