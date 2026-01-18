import pytest
import cartopy.feature as cfeature
def test_change_scale(self):
    new_lakes = cfeature.LAKES.with_scale('10m')
    assert new_lakes.scale == '10m'
    assert new_lakes.kwargs == cfeature.LAKES.kwargs
    assert new_lakes.category == cfeature.LAKES.category
    assert new_lakes.name == cfeature.LAKES.name