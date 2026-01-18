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
def percy_snapshot(self, name='', wait_for_callbacks=False, convert_canvases=False, widths=None):
    """percy_snapshot - visual test api shortcut to `percy_runner.snapshot`.
        It also combines the snapshot `name` with the Python version,
        args:
        - name: combined with the python version to give the final snapshot name
        - wait_for_callbacks: default False, whether to wait for Dash callbacks,
            after an extra second to ensure that any relevant callbacks have
            been initiated
        - convert_canvases: default False, whether to convert all canvas elements
            in the DOM into static images for percy to see. They will be restored
            after the snapshot is complete.
        - widths: a list of pixel widths for percy to render the page with. Note
            that this does not change the browser in which the DOM is constructed,
            so the width will only affect CSS, not JS-driven layout.
            Defaults to [1280]
        """
    if widths is None:
        widths = [1280]
    logger.info('taking snapshot name => %s', name)
    try:
        if wait_for_callbacks:
            time.sleep(1)
            until(self._wait_for_callbacks, timeout=40, poll=0.3)
    except TestingTimeoutError:
        logger.error('wait_for_callbacks failed => status of invalid rqs %s', self.redux_state_rqs)
    if convert_canvases:
        self.driver.execute_script("\n                const stash = window._canvasStash = [];\n                Array.from(document.querySelectorAll('canvas')).forEach(c => {\n                    const i = document.createElement('img');\n                    i.src = c.toDataURL();\n                    i.width = c.width;\n                    i.height = c.height;\n                    i.setAttribute('style', c.getAttribute('style'));\n                    i.className = c.className;\n                    i.setAttribute('data-canvasnum', stash.length);\n                    stash.push(c);\n                    c.parentElement.insertBefore(i, c);\n                    c.parentElement.removeChild(c);\n                });\n            ")
    try:
        self.percy_runner.snapshot(name=name, widths=widths)
    except requests.HTTPError as err:
        if err.request.status_code != 400:
            raise err
    if convert_canvases:
        self.driver.execute_script("\n                const stash = window._canvasStash;\n                Array.from(\n                    document.querySelectorAll('img[data-canvasnum]')\n                ).forEach(i => {\n                    const c = stash[+i.getAttribute('data-canvasnum')];\n                    i.parentElement.insertBefore(c, i);\n                    i.parentElement.removeChild(i);\n                });\n                delete window._canvasStash;\n            ")