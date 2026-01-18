from typing import List
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import UnexpectedTagNameException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
def select_by_visible_text(self, text: str) -> None:
    """Select all options that display text matching the argument. That is,
        when given "Bar" this would select an option like:

         <option value="foo">Bar</option>

        :Args:
         - text - The visible text to match against

         throws NoSuchElementException If there is no option with specified text in SELECT
        """
    xpath = f'.//option[normalize-space(.) = {self._escape_string(text)}]'
    opts = self._el.find_elements(By.XPATH, xpath)
    matched = False
    for opt in opts:
        self._set_selected(opt)
        if not self.is_multiple:
            return
        matched = True
    if len(opts) == 0 and ' ' in text:
        sub_string_without_space = self._get_longest_token(text)
        if sub_string_without_space == '':
            candidates = self.options
        else:
            xpath = f'.//option[contains(.,{self._escape_string(sub_string_without_space)})]'
            candidates = self._el.find_elements(By.XPATH, xpath)
        for candidate in candidates:
            if text == candidate.text:
                self._set_selected(candidate)
                if not self.is_multiple:
                    return
                matched = True
    if not matched:
        raise NoSuchElementException(f'Could not locate element with visible text: {text}')