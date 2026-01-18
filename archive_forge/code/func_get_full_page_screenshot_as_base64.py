import base64
import logging
import os
import warnings
import zipfile
from contextlib import contextmanager
from io import BytesIO
from selenium.webdriver.common.driver_finder import DriverFinder
from selenium.webdriver.remote.webdriver import WebDriver as RemoteWebDriver
from .options import Options
from .remote_connection import FirefoxRemoteConnection
from .service import Service
def get_full_page_screenshot_as_base64(self) -> str:
    """Gets the full document screenshot of the current window as a base64
        encoded string which is useful in embedded images in HTML.

        :Usage:
            ::

                driver.get_full_page_screenshot_as_base64()
        """
    return self.execute('FULL_PAGE_SCREENSHOT')['value']