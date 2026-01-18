from __future__ import annotations
import asyncio
from typing import Optional, Dict, Any, List, Union, Type, TYPE_CHECKING
from .base import BaseGlobalClient
@property
def context_manager(self) -> 'PlaywrightContextManager':
    """
        Returns the Playwright context manager
        """
    if self._pw_context_manager is None:
        from playwright.sync_api import sync_playwright
        self._pw_context_manager = sync_playwright()
    return self._pw_context_manager