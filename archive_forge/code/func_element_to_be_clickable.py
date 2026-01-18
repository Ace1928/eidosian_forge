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
def element_to_be_clickable(mark: Union[WebElement, Tuple[str, str]]) -> Callable[[WebDriverOrWebElement], Union[Literal[False], WebElement]]:
    """An Expectation for checking an element is visible and enabled such that
    you can click it.

    element is either a locator (text) or an WebElement
    """

    def _predicate(driver: WebDriverOrWebElement):
        target = mark
        if not isinstance(target, WebElement):
            target = driver.find_element(*target)
        element = visibility_of(target)(driver)
        if element and element.is_enabled():
            return element
        return False
    return _predicate