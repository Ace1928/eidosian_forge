from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, Optional
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.utils import get_from_env
def import_langkit(sentiment: bool=False, toxicity: bool=False, themes: bool=False) -> Any:
    """Import the langkit python package and raise an error if it is not installed.

    Args:
        sentiment: Whether to import the langkit.sentiment module. Defaults to False.
        toxicity: Whether to import the langkit.toxicity module. Defaults to False.
        themes: Whether to import the langkit.themes module. Defaults to False.

    Returns:
        The imported langkit module.
    """
    try:
        import langkit
        import langkit.regexes
        import langkit.textstat
        if sentiment:
            import langkit.sentiment
        if toxicity:
            import langkit.toxicity
        if themes:
            import langkit.themes
    except ImportError:
        raise ImportError('To use the whylabs callback manager you need to have the `langkit` python package installed. Please install it with `pip install langkit`.')
    return langkit