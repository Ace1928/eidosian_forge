from __future__ import annotations
import asyncio
from typing import TYPE_CHECKING, Any, Coroutine, List, Optional, TypeVar
def create_sync_playwright_browser(headless: bool=True, args: Optional[List[str]]=None) -> SyncBrowser:
    """
    Create a playwright browser.

    Args:
        headless: Whether to run the browser in headless mode. Defaults to True.
        args: arguments to pass to browser.chromium.launch

    Returns:
        SyncBrowser: The playwright browser.
    """
    from playwright.sync_api import sync_playwright
    browser = sync_playwright().start()
    return browser.chromium.launch(headless=headless, args=args)