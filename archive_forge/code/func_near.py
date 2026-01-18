from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
def near(self, element_or_locator_distance: Union[WebElement, Dict, int]=None) -> 'RelativeBy':
    """Add a filter to look for elements near.

        :Args:
            - element_or_locator_distance: Element to look near by the element or within a distance
        """
    if not element_or_locator_distance:
        raise WebDriverException('Element or locator or distance must be given when calling near method')
    self.filters.append({'kind': 'near', 'args': [element_or_locator_distance]})
    return self