import re
from collections.abc import Iterable
from typing import Any
from typing import Callable
from typing import List
from typing import Literal
from typing import Tuple
from typing import TypeVar
from typing import Union
from selenium.common.exceptions import NoAlertPresentException
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import NoSuchFrameException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webdriver import WebElement
def element_located_selection_state_to_be(locator: Tuple[str, str], is_selected: bool) -> Callable[[WebDriverOrWebElement], bool]:
    """An expectation to locate an element and check if the selection state
    specified is in that state.

    locator is a tuple of (by, path) is_selected is a boolean
    """

    def _predicate(driver: WebDriverOrWebElement):
        try:
            element = driver.find_element(*locator)
            return element.is_selected() == is_selected
        except StaleElementReferenceException:
            return False
    return _predicate