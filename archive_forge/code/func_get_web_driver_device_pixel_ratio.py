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
def get_web_driver_device_pixel_ratio(web_driver: WebDriver) -> float:
    calculate_web_driver_device_pixel_ratio = '        return window.devicePixelRatio\n    '
    device_pixel_ratio: float = web_driver.execute_script(calculate_web_driver_device_pixel_ratio)
    return device_pixel_ratio