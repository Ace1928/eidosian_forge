import sys
from typing import TYPE_CHECKING, Any, Dict
def register_application_for_autosummary(app: 'Sphinx') -> None:
    """Register application object to autosummary module.

    Since Sphinx-1.7, documenters and attrgetters are registered into
    application object.  As a result, the arguments of
    ``get_documenter()`` has been changed.  To keep compatibility,
    this handler registers application object to the module.
    """
    if 'sphinx.ext.autosummary' in sys.modules:
        from sphinx.ext import autosummary
        if hasattr(autosummary, '_objects'):
            autosummary._objects['_app'] = app
        else:
            autosummary._app = app