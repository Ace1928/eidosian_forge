from typing import List
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import UnexpectedTagNameException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
def select_by_value(self, value: str) -> None:
    """Select all options that have a value matching the argument. That is,
        when given "foo" this would select an option like:

        <option value="foo">Bar</option>

        :Args:
         - value - The value to match against

        throws NoSuchElementException If there is no option with specified value in SELECT
        """
    css = f'option[value ={self._escape_string(value)}]'
    opts = self._el.find_elements(By.CSS_SELECTOR, css)
    matched = False
    for opt in opts:
        self._set_selected(opt)
        if not self.is_multiple:
            return
        matched = True
    if not matched:
        raise NoSuchElementException(f'Cannot locate option with value: {value}')