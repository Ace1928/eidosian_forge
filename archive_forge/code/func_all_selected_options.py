from typing import List
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import UnexpectedTagNameException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
@property
def all_selected_options(self) -> List[WebElement]:
    """Returns a list of all selected options belonging to this select
        tag."""
    return [opt for opt in self.options if opt.is_selected()]