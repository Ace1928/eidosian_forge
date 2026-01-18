import typing
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.options import ArgOptions
@binary_location.setter
def binary_location(self, value: str) -> None:
    """Allows you to set the browser binary to launch.

        :Args:
         - value : path to the browser binary
        """
    if not isinstance(value, str):
        raise TypeError(self.BINARY_LOCATION_ERROR)
    self._binary_location = value