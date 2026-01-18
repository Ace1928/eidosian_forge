from __future__ import unicode_literals
import re
import pytest
import pybtex.database.input.bibtex
import pybtex.plugin
import pybtex.style.formatting.plain
def test_bad_find_plugin():
    with pytest.raises(pybtex.plugin.PluginGroupNotFound):
        pybtex.plugin.find_plugin('pybtex.invalid.group', '__oops')
    with pytest.raises(pybtex.plugin.PluginNotFound) as excinfo:
        pybtex.plugin.find_plugin('pybtex.style.formatting', '__oops')
    assert 'plugin pybtex.style.formatting.__oops not found' in str(excinfo.value)
    with pytest.raises(pybtex.plugin.PluginNotFound):
        pybtex.plugin.find_plugin('pybtex.style.formatting', filename='oh.__oops')