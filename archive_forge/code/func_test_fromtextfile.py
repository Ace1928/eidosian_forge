import numpy as np
import numpy.ma as ma
from numpy import recarray
from numpy.ma import masked, nomask
from numpy.testing import temppath
from numpy.core.records import (
from numpy.ma.mrecords import (
from numpy.ma.testutils import (
from numpy.compat import pickle
def test_fromtextfile(self):
    fcontent = '#\n\'One (S)\',\'Two (I)\',\'Three (F)\',\'Four (M)\',\'Five (-)\',\'Six (C)\'\n\'strings\',1,1.0,\'mixed column\',,1\n\'with embedded "double quotes"\',2,2.0,1.0,,1\n\'strings\',3,3.0E5,3,,1\n\'strings\',4,-1e-10,,,1\n'
    with temppath() as path:
        with open(path, 'w') as f:
            f.write(fcontent)
        mrectxt = fromtextfile(path, delimiter=',', varnames='ABCDEFG')
    assert_(isinstance(mrectxt, MaskedRecords))
    assert_equal(mrectxt.F, [1, 1, 1, 1])
    assert_equal(mrectxt.E._mask, [1, 1, 1, 1])
    assert_equal(mrectxt.C, [1, 2, 300000.0, -1e-10])