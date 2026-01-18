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
def presence_of_all_elements_located(locator: Tuple[str, str]) -> Callable[[WebDriverOrWebElement], List[WebElement]]:
    """An expectation for checking that there is at least one element present
    on a web page.

    locator is used to find the element returns the list of WebElements
    once they are located
    """

    def _predicate(driver: WebDriverOrWebElement):
        return driver.find_elements(*locator)
    return _predicate