import os
import warnings
import pkgutil
from plotly.optional_imports import get_module
from plotly import tools
from ._plotlyjs_version import __plotlyjs_version__
def get_plotlyjs_version():
    """
    Returns the version of plotly.js that is bundled with plotly.py.

    Returns
    -------
    str
        Plotly.js version string
    """
    return __plotlyjs_version__