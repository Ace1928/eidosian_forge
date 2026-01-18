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
def click_at_coord_fractions(self, elem_or_selector, fx, fy):
    elem = self._get_element(elem_or_selector)
    ActionChains(self.driver).move_to_element_with_offset(elem, elem.size['width'] * fx, elem.size['height'] * fy).click().perform()