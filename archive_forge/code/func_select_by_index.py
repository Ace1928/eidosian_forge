from typing import List
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import UnexpectedTagNameException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
def select_by_index(self, index: int) -> None:
    """Select the option at the given index. This is done by examining the
        "index" attribute of an element, and not merely by counting.

        :Args:
         - index - The option at this index will be selected

        throws NoSuchElementException If there is no option with specified index in SELECT
        """
    match = str(index)
    for opt in self.options:
        if opt.get_attribute('index') == match:
            self._set_selected(opt)
            return
    raise NoSuchElementException(f'Could not locate element with index {index}')