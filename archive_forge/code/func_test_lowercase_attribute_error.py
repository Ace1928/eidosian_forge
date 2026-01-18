import pytest
from holoviews.core.element import AttrTree
from holoviews.element.comparison import ComparisonTestCase
def test_lowercase_attribute_error(self):
    msg = "'AttrTree' object has no attribute c\\."
    with pytest.raises(AttributeError, match=msg):
        self.tree.c