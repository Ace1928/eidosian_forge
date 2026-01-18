from __future__ import annotations
from typing import Optional, Type
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_community.tools.playwright.base import BaseBrowserTool
from langchain_community.tools.playwright.utils import (
@root_validator
def check_acheck_bs_importrgs(cls, values: dict) -> dict:
    """Check that the arguments are valid."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("The 'beautifulsoup4' package is required to use this tool. Please install it with 'pip install beautifulsoup4'.")
    return values