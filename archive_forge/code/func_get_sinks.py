from selenium.webdriver.chromium.remote_connection import ChromiumRemoteConnection
from selenium.webdriver.common.driver_finder import DriverFinder
from selenium.webdriver.common.options import ArgOptions
from selenium.webdriver.common.service import Service
from selenium.webdriver.remote.webdriver import WebDriver as RemoteWebDriver
def get_sinks(self) -> list:
    """:Returns: A list of sinks available for Cast."""
    return self.execute('getSinks')['value']