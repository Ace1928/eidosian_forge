from selenium.webdriver.chromium.remote_connection import ChromiumRemoteConnection
from selenium.webdriver.common.driver_finder import DriverFinder
from selenium.webdriver.common.options import ArgOptions
from selenium.webdriver.common.service import Service
from selenium.webdriver.remote.webdriver import WebDriver as RemoteWebDriver
def get_network_conditions(self):
    """Gets Chromium network emulation settings.

        :Returns:     A dict. For example:     {'latency': 4,
        'download_throughput': 2, 'upload_throughput': 2,     'offline':
        False}
        """
    return self.execute('getNetworkConditions')['value']