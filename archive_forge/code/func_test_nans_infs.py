import platform
import pytest
import numpy as np
from numpy import uint16, float16, float32, float64
from numpy.testing import assert_, assert_equal, _OLD_PROMOTION, IS_WASM
def test_nans_infs(self):
    with np.errstate(all='ignore'):
        assert_equal(np.isnan(self.all_f16), np.isnan(self.all_f32))
        assert_equal(np.isinf(self.all_f16), np.isinf(self.all_f32))
        assert_equal(np.isfinite(self.all_f16), np.isfinite(self.all_f32))
        assert_equal(np.signbit(self.all_f16), np.signbit(self.all_f32))
        assert_equal(np.spacing(float16(65504)), np.inf)
        nan = float16(np.nan)
        assert_(not (self.all_f16 == nan).any())
        assert_(not (nan == self.all_f16).any())
        assert_((self.all_f16 != nan).all())
        assert_((nan != self.all_f16).all())
        assert_(not (self.all_f16 < nan).any())
        assert_(not (nan < self.all_f16).any())
        assert_(not (self.all_f16 <= nan).any())
        assert_(not (nan <= self.all_f16).any())
        assert_(not (self.all_f16 > nan).any())
        assert_(not (nan > self.all_f16).any())
        assert_(not (self.all_f16 >= nan).any())
        assert_(not (nan >= self.all_f16).any())