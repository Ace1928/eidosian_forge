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
def is_displayed(self) -> bool:
    """Whether the element is visible to a user."""
    if isDisplayed_js is None:
        _load_js()
    return self.parent.execute_script(f'/* isDisplayed */return ({isDisplayed_js}).apply(null, arguments);', self)