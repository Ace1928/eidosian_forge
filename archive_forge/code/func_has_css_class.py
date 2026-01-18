from contextlib import contextmanager
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from django.test import modify_settings
from django.test.selenium import SeleniumTestCase
from django.utils.deprecation import MiddlewareMixin
from django.utils.translation import gettext as _
def has_css_class(self, selector, klass):
    """
        Return True if the element identified by `selector` has the CSS class
        `klass`.
        """
    from selenium.webdriver.common.by import By
    return self.selenium.find_element(By.CSS_SELECTOR, selector).get_attribute('class').find(klass) != -1