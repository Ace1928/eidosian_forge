import logging
import sys
import warnings
from typing import Optional
import wandb
def display_widget(widget):
    """Display ipywidgets in notebooks, is a noop outside of a jupyter context."""
    if wandb.run and wandb.run._settings.silent:
        return
    try:
        from IPython.core.display import display
    except ImportError:
        wandb.termwarn("Unable to render Widget, can't import display from ipython.core")
        return False
    return display(widget)