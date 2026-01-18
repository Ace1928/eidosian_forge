from contextlib import contextmanager
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from django.test import modify_settings
from django.test.selenium import SeleniumTestCase
from django.utils.deprecation import MiddlewareMixin
from django.utils.translation import gettext as _
def assertCountSeleniumElements(self, selector, count, root_element=None):
    """
        Assert number of matches for a CSS selector.

        `root_element` allow restriction to a pre-selected node.
        """
    from selenium.webdriver.common.by import By
    root_element = root_element or self.selenium
    self.assertEqual(len(root_element.find_elements(By.CSS_SELECTOR, selector)), count)