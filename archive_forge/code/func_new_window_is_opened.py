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
def new_window_is_opened(current_handles: List[str]) -> Callable[[WebDriver], bool]:
    """An expectation that a new window will be opened and have the number of
    windows handles increase."""

    def _predicate(driver: WebDriver):
        return len(driver.window_handles) > len(current_handles)
    return _predicate