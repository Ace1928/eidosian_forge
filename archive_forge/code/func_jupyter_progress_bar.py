import logging
import sys
import warnings
from typing import Optional
import wandb
def jupyter_progress_bar(min: float=0, max: float=1.0) -> Optional[ProgressWidget]:
    """Return an ipywidget progress bar or None if we can't import it."""
    widgets = wandb.util.get_module('ipywidgets')
    try:
        if widgets is None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                from IPython.html import widgets
        assert hasattr(widgets, 'VBox')
        assert hasattr(widgets, 'Label')
        assert hasattr(widgets, 'FloatProgress')
        return ProgressWidget(widgets, min=min, max=max)
    except (ImportError, AssertionError):
        return None