import os
import pytest
from lxml import etree
import json
import tempfile
@pytest.fixture
def nb_metadata(defaultCellHeight, cellMargin, maxColumns, activeView):
    grid_default = {'name': 'grid', 'type': 'grid'}
    if defaultCellHeight is not None:
        grid_default['defaultCellHeight'] = defaultCellHeight
    if cellMargin is not None:
        grid_default['cellMargin'] = cellMargin
    if maxColumns is not None:
        grid_default['maxColumns'] = maxColumns
    activeView_dict = {'activeView': activeView} if activeView else {}
    return {'extensions': {'jupyter_dashboards': {'views': {'grid_default': grid_default}, **activeView_dict}}}