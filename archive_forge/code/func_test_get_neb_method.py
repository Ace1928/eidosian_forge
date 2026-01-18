from pytest import warns, raises
from ase import Atoms
from ase import neb
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
def test_get_neb_method():
    neb_dummy = neb.NEB([])
    assert isinstance(neb.get_neb_method(neb_dummy, 'eb'), neb.FullSpringMethod)
    assert isinstance(neb.get_neb_method(neb_dummy, 'aseneb'), neb.ASENEBMethod)
    assert isinstance(neb.get_neb_method(neb_dummy, 'improvedtangent'), neb.ImprovedTangentMethod)
    assert isinstance(neb.get_neb_method(neb_dummy, 'spline'), neb.SplineMethod)
    assert isinstance(neb.get_neb_method(neb_dummy, 'string'), neb.StringMethod)
    with raises(ValueError, match='.*some_random_string.*'):
        _ = neb.get_neb_method(neb_dummy, 'some_random_string')