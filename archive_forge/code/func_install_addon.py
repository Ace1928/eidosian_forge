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
def install_addon(self, path, temporary=False) -> str:
    """Installs Firefox addon.

        Returns identifier of installed addon. This identifier can later
        be used to uninstall addon.

        :param temporary: allows you to load browser extensions temporarily during a session
        :param path: Absolute path to the addon that will be installed.

        :Usage:
            ::

                driver.install_addon('/path/to/firebug.xpi')
        """
    if os.path.isdir(path):
        fp = BytesIO()
        path = os.path.normpath(path)
        path_root = len(path) + 1
        with zipfile.ZipFile(fp, 'w', zipfile.ZIP_DEFLATED) as zipped:
            for base, _, files in os.walk(path):
                for fyle in files:
                    filename = os.path.join(base, fyle)
                    zipped.write(filename, filename[path_root:])
        addon = base64.b64encode(fp.getvalue()).decode('UTF-8')
    else:
        with open(path, 'rb') as file:
            addon = base64.b64encode(file.read()).decode('UTF-8')
    payload = {'addon': addon, 'temporary': temporary}
    return self.execute('INSTALL_ADDON', payload)['value']