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
def frame_to_be_available_and_switch_to_it(locator: Union[Tuple[str, str], str]) -> Callable[[WebDriver], bool]:
    """An expectation for checking whether the given frame is available to
    switch to.

    If the frame is available it switches the given driver to the
    specified frame.
    """

    def _predicate(driver: WebDriver):
        try:
            if isinstance(locator, Iterable) and (not isinstance(locator, str)):
                driver.switch_to.frame(driver.find_element(*locator))
            else:
                driver.switch_to.frame(locator)
            return True
        except NoSuchFrameException:
            return False
    return _predicate