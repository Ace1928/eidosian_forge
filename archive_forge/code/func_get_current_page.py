from __future__ import annotations
import asyncio
from typing import TYPE_CHECKING, Any, Coroutine, List, Optional, TypeVar
def get_current_page(browser: SyncBrowser) -> SyncPage:
    """
    Get the current page of the browser.
    Args:
        browser: The browser to get the current page from.

    Returns:
        SyncPage: The current page.
    """
    if not browser.contexts:
        context = browser.new_context()
        return context.new_page()
    context = browser.contexts[0]
    if not context.pages:
        return context.new_page()
    return context.pages[-1]