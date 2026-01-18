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
def create_chromium_webdriver(extra_options: list[str] | None=None, scale_factor: float=1) -> WebDriver:
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.webdriver import WebDriver as Chrome
    executable_path = settings.chromedriver_path()
    if executable_path is None:
        for executable in ['chromedriver', 'chromium.chromedriver', 'chromedriver-binary']:
            executable_path = which(executable)
            if executable_path is not None:
                break
        else:
            raise RuntimeError("chromedriver or its variant is not installed or not present on PATH; use BOKEH_CHROMEDRIVER_PATH to specify a customized chromedriver's location")
    service = Service(executable_path)
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--hide-scrollbars')
    options.add_argument(f'--force-device-scale-factor={scale_factor}')
    options.add_argument('--force-color-profile=srgb')
    if extra_options:
        for op in extra_options:
            options.add_argument(op)
    if os.getenv('BOKEH_IN_DOCKER') == '1':
        options.add_argument('--no-sandbox')
    return Chrome(service=service, options=options)