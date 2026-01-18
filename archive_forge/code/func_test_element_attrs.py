import os.path as op
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
@skip_if_no_network
def test_element_attrs():
    assert isinstance(central.xpath.element_attrs('xnat:voxelRes'), list)
    assert {'x', 'y', 'z'}.issubset(central.xpath.element_keys('xnat:voxelRes'))
    assert '0.9375' in central.xpath.element_values('xnat:voxelRes', 'x')