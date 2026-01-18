from __future__ import annotations
import os
import pkgutil
import warnings
import zipfile
from abc import ABCMeta
from base64 import b64decode
from base64 import encodebytes
from hashlib import md5 as md5_hash
from io import BytesIO
from typing import List
from selenium.common.exceptions import JavascriptException
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.utils import keys_to_typing
from .command import Command
from .shadowroot import ShadowRoot
@property
def location_once_scrolled_into_view(self) -> dict:
    """THIS PROPERTY MAY CHANGE WITHOUT WARNING. Use this to discover where
        on the screen an element is so that we can click it. This method should
        cause the element to be scrolled into view.

        Returns the top lefthand corner location on the screen, or zero
        coordinates if the element is not visible.
        """
    old_loc = self._execute(Command.W3C_EXECUTE_SCRIPT, {'script': 'arguments[0].scrollIntoView(true); return arguments[0].getBoundingClientRect()', 'args': [self]})['value']
    return {'x': round(old_loc['x']), 'y': round(old_loc['y'])}