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
def staleness_of(element: WebElement) -> Callable[[Any], bool]:
    """Wait until an element is no longer attached to the DOM.

    element is the element to wait for. returns False if the element is
    still attached to the DOM, true otherwise.
    """

    def _predicate(_):
        try:
            element.is_enabled()
            return False
        except StaleElementReferenceException:
            return True
    return _predicate