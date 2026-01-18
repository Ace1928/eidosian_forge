import os
import pytest
from lxml import etree
@pytest.fixture(params=[False, True], ids='show_handles={}'.format)
def show_handles(request):
    return request.param