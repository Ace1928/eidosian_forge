import time
from http.client import HTTPConnection
from subprocess import PIPE, Popen
import pytest
from playwright.sync_api import expect
def test_jupyterlite_execution(launch_jupyterlite, page):
    page.goto('http://localhost:8123/index.html')
    page.locator('text="Getting_Started.ipynb"').first.dblclick()
    if page.locator('.jp-Dialog').count() == 1:
        page.locator('.jp-select-wrapper > select').select_option('Python (Pyodide)')
        page.locator('.jp-Dialog-footer > button').nth(1).click()
    for _ in range(6):
        page.locator('button[data-command="notebook:run-cell-and-select-next"]').click()
        page.wait_for_timeout(500)
    page.locator('.noUi-handle').click(timeout=120 * 1000)
    page.keyboard.press('ArrowRight')
    expect(page.locator('.bk-panel-models-markup-HTML').locator('div').locator('pre')).to_have_text('0.1')