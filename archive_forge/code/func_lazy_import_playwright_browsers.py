from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple, Type
from langchain_core.pydantic_v1 import root_validator
from langchain_core.tools import BaseTool
def lazy_import_playwright_browsers() -> Tuple[Type[AsyncBrowser], Type[SyncBrowser]]:
    """
    Lazy import playwright browsers.

    Returns:
        Tuple[Type[AsyncBrowser], Type[SyncBrowser]]:
            AsyncBrowser and SyncBrowser classes.
    """
    try:
        from playwright.async_api import Browser as AsyncBrowser
        from playwright.sync_api import Browser as SyncBrowser
    except ImportError:
        raise ImportError("The 'playwright' package is required to use the playwright tools. Please install it with 'pip install playwright'.")
    return (AsyncBrowser, SyncBrowser)