from typing import List
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import UnexpectedTagNameException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
@property
def first_selected_option(self) -> WebElement:
    """The first selected option in this select tag (or the currently
        selected option in a normal select)"""
    for opt in self.options:
        if opt.is_selected():
            return opt
    raise NoSuchElementException('No options are selected')