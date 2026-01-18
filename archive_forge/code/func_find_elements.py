import os
import sys
import time
import logging
import warnings
import percy
import requests
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import (
from dash.testing.wait import (
from dash.testing.dash_page import DashPageMixin
from dash.testing.errors import DashAppLoadingError, BrowserError, TestingTimeoutError
from dash.testing.consts import SELENIUM_GRID_DEFAULT
def find_elements(self, selector, attribute='CSS_SELECTOR'):
    """find_elements returns a list of all elements matching the attribute
        `selector`. Shortcut to `driver.find_elements(By.CSS_SELECTOR, ...)`.
        args:
        - attribute: the attribute type to search for, aligns with the Selenium
            API's `By` class. default "CSS_SELECTOR"
            valid values: "CSS_SELECTOR", "ID", "NAME", "TAG_NAME",
            "CLASS_NAME", "LINK_TEXT", "PARTIAL_LINK_TEXT", "XPATH"
        """
    return self.driver.find_elements(getattr(By, attribute.upper()), selector)