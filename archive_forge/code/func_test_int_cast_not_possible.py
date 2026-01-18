import numpy as np
from skimage._shared.utils import safe_as_int
from skimage._shared import testing
def test_int_cast_not_possible():
    with testing.raises(ValueError):
        safe_as_int(7.1)
    with testing.raises(ValueError):
        safe_as_int([7.1, 0.9])
    with testing.raises(ValueError):
        safe_as_int(np.r_[7.1, 0.9])
    with testing.raises(ValueError):
        safe_as_int((7.1, 0.9))
    with testing.raises(ValueError):
        safe_as_int(((3, 4, 1), (2, 7.6, 289)))
    with testing.raises(ValueError):
        safe_as_int(7.1, 0.09)
    with testing.raises(ValueError):
        safe_as_int([7.1, 0.9], 0.09)
    with testing.raises(ValueError):
        safe_as_int(np.r_[7.1, 0.9], 0.09)
    with testing.raises(ValueError):
        safe_as_int((7.1, 0.9), 0.09)
    with testing.raises(ValueError):
        safe_as_int(((3, 4, 1), (2, 7.6, 289)), 0.25)