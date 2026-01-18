import pytest
from panel import Column, GridBox, Spacer
from panel.tests.util import serve_component, wait_until
def test_gridbox(page):
    grid = GridBox(*(Spacer(width=50, height=50) for i in range(12)), ncols=4)
    serve_component(page, grid)
    wait_until(lambda: page.locator('.bk-GridBox').bounding_box() == {'width': 200, 'height': 150, 'x': 0, 'y': 0}, page)
    grid.ncols = 6
    wait_until(lambda: page.locator('.bk-GridBox').bounding_box() == {'width': 300, 'height': 100, 'x': 0, 'y': 0}, page)