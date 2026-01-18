import logging
import sys
import warnings
from typing import Optional
import wandb
def display_html(html: str):
    """Display HTML in notebooks, is a noop outside a jupyter context."""
    if wandb.run and wandb.run._settings.silent:
        return
    try:
        from IPython.core.display import HTML, display
    except ImportError:
        wandb.termwarn("Unable to render HTML, can't import display from ipython.core")
        return False
    return display(HTML(html))