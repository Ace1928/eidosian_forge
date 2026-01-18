from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.options import ArgOptions
@overlay_scrollbars_enabled.setter
def overlay_scrollbars_enabled(self, value) -> None:
    """Allows you to enable or disable overlay scrollbars.

        :Args:
         - value : True or False
        """
    self._overlay_scrollbars_enabled = value