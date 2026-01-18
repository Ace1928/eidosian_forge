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
def get_full_page_screenshot_as_file(self, filename) -> bool:
    """Saves a full document screenshot of the current window to a PNG
        image file. Returns False if there is any IOError, else returns True.
        Use full paths in your filename.

        :Args:
         - filename: The full path you wish to save your screenshot to. This
           should end with a `.png` extension.

        :Usage:
            ::

                driver.get_full_page_screenshot_as_file('/Screenshots/foo.png')
        """
    if not filename.lower().endswith('.png'):
        warnings.warn('name used for saved screenshot does not match file type. It should end with a `.png` extension', UserWarning)
    png = self.get_full_page_screenshot_as_png()
    try:
        with open(filename, 'wb') as f:
            f.write(png)
    except OSError:
        return False
    finally:
        del png
    return True