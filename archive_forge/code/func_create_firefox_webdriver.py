from __future__ import annotations
import logging # isort:skip
from ..util.dependencies import import_required # isort:skip
import_required("selenium.webdriver",
import atexit
import os
from os.path import devnull
from shutil import which
from typing import TYPE_CHECKING, Literal
from packaging.version import Version
from ..settings import settings
def create_firefox_webdriver(scale_factor: float=1) -> WebDriver:
    firefox = which('firefox')
    if firefox is None:
        raise RuntimeError('firefox is not installed or not present on PATH')
    geckodriver = which('geckodriver')
    if geckodriver is None:
        raise RuntimeError('geckodriver is not installed or not present on PATH')
    import selenium
    from selenium.webdriver.firefox.options import Options
    from selenium.webdriver.firefox.service import Service
    from selenium.webdriver.firefox.webdriver import WebDriver as Firefox
    if Version(selenium.__version__) >= Version('4.11'):
        service = Service()
    else:
        service = Service(log_path=devnull)
    options = Options()
    options.add_argument('--headless')
    options.set_preference('layout.css.devPixelsPerPx', f'{scale_factor}')
    return Firefox(service=service, options=options)