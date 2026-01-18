from __future__ import annotations
import asyncio
from typing import TYPE_CHECKING, Any, Coroutine, List, Optional, TypeVar
def create_async_playwright_browser(headless: bool=True, args: Optional[List[str]]=None) -> AsyncBrowser:
    """
    Create an async playwright browser.

    Args:
        headless: Whether to run the browser in headless mode. Defaults to True.
        args: arguments to pass to browser.chromium.launch

    Returns:
        AsyncBrowser: The playwright browser.
    """
    from playwright.async_api import async_playwright
    browser = run_async(async_playwright().start())
    return run_async(browser.chromium.launch(headless=headless, args=args))