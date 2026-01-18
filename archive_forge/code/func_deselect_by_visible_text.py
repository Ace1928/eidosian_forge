from typing import List
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import UnexpectedTagNameException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
def deselect_by_visible_text(self, text: str) -> None:
    """Deselect all options that display text matching the argument. That
        is, when given "Bar" this would deselect an option like:

        <option value="foo">Bar</option>

        :Args:
         - text - The visible text to match against
        """
    if not self.is_multiple:
        raise NotImplementedError('You may only deselect options of a multi-select')
    matched = False
    xpath = f'.//option[normalize-space(.) = {self._escape_string(text)}]'
    opts = self._el.find_elements(By.XPATH, xpath)
    for opt in opts:
        self._unset_selected(opt)
        matched = True
    if not matched:
        raise NoSuchElementException(f'Could not locate element with visible text: {text}')