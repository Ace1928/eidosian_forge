from typing import Optional
from typing import Union
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import NoSuchFrameException
from selenium.common.exceptions import NoSuchWindowException
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from .command import Command
def new_window(self, type_hint: Optional[str]=None) -> None:
    """Switches to a new top-level browsing context.

        The type hint can be one of "tab" or "window". If not specified the
        browser will automatically select it.

        :Usage:
            ::

                driver.switch_to.new_window('tab')
        """
    value = self._driver.execute(Command.NEW_WINDOW, {'type': type_hint})['value']
    self._w3c_window(value['handle'])