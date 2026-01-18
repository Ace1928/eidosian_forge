import json
import re
from .widgets import Widget, DOMWidget, widget as widget_module
from .widgets.widget_link import Link
from .widgets.docutils import doc_subst
from ._version import __html_manager_version__
def dependency_state(widgets, drop_defaults=True):
    """Get the state of all widgets specified, and their dependencies.

    This uses a simple dependency finder, including:
     - any widget directly referenced in the state of an included widget
     - any widget in a list/tuple attribute in the state of an included widget
     - any widget in a dict attribute in the state of an included widget
     - any jslink/jsdlink between two included widgets
    What this alogorithm does not do:
     - Find widget references in nested list/dict structures
     - Find widget references in other types of attributes

    Note that this searches the state of the widgets for references, so if
    a widget reference is not included in the serialized state, it won't
    be considered as a dependency.

    Parameters
    ----------
    widgets: single widget or list of widgets.
       This function will return the state of every widget mentioned
       and of all their dependencies.
    drop_defaults: boolean
        Whether to drop default values from the widget states.

    Returns
    -------
    A dictionary with the state of the widgets and any widget they
    depend on.
    """
    if widgets is None:
        state = Widget.get_manager_state(drop_defaults=drop_defaults, widgets=None)['state']
    else:
        try:
            widgets[0]
        except (IndexError, TypeError):
            widgets = [widgets]
        state = {}
        for widget in widgets:
            _get_recursive_state(widget, state, drop_defaults)
        add_resolved_links(state, drop_defaults)
    return state