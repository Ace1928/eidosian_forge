from contextlib import contextmanager
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from django.test import modify_settings
from django.test.selenium import SeleniumTestCase
from django.utils.deprecation import MiddlewareMixin
from django.utils.translation import gettext as _
def deselect_option(self, selector, value):
    """
        Deselect the <OPTION> with the value `value` inside the <SELECT> widget
        identified by the CSS selector `selector`.
        """
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import Select
    select = Select(self.selenium.find_element(By.CSS_SELECTOR, selector))
    select.deselect_by_value(value)