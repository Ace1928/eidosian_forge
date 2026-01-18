from selenium.common.exceptions import WebDriverException
from selenium.webdriver.remote.webdriver import WebDriver as RemoteWebDriver
from ..common.driver_finder import DriverFinder
from .options import Options
from .remote_connection import SafariRemoteConnection
from .service import Service
def get_permission(self, permission):
    payload = self.execute('GET_PERMISSIONS')['value']
    permissions = payload['permissions']
    if not permissions:
        return None
    if permission not in permissions:
        return None
    value = permissions[permission]
    if not isinstance(value, bool):
        return None
    return value