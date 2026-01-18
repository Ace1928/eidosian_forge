from datetime import datetime
import sys
import numpy as np
import pytest
from pandas.compat import PYPY
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.accessor import PandasDelegate
from pandas.core.base import (
def test_invalid_delegation(self):
    self.Delegate._add_delegate_accessors(delegate=self.Delegator, accessors=self.Delegator._properties, typ='property')
    self.Delegate._add_delegate_accessors(delegate=self.Delegator, accessors=self.Delegator._methods, typ='method')
    delegate = self.Delegate(self.Delegator())
    msg = 'You cannot access the property prop'
    with pytest.raises(TypeError, match=msg):
        delegate.prop
    msg = 'The property prop cannot be set'
    with pytest.raises(TypeError, match=msg):
        delegate.prop = 5
    msg = 'You cannot access the property prop'
    with pytest.raises(TypeError, match=msg):
        delegate.prop