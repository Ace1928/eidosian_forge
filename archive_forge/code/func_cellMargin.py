import os
import pytest
from lxml import etree
import json
import tempfile
@pytest.fixture(params=[None, 0, 10], ids='cellMargin={}'.format)
def cellMargin(request):
    return request.param