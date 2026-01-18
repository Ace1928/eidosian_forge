import pytest
import cartopy.feature as cfeature
def test_bad_ne_scale():
    with pytest.raises(ValueError, match='not a valid Natural Earth scale'):
        cfeature.NaturalEarthFeature('physical', 'land', '30m')