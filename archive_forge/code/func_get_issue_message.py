from selenium.webdriver.chromium.remote_connection import ChromiumRemoteConnection
from selenium.webdriver.common.driver_finder import DriverFinder
from selenium.webdriver.common.options import ArgOptions
from selenium.webdriver.common.service import Service
from selenium.webdriver.remote.webdriver import WebDriver as RemoteWebDriver
def get_issue_message(self):
    """:Returns: An error message when there is any issue in a Cast
        session."""
    return self.execute('getIssueMessage')['value']