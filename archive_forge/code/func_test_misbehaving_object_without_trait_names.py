from IPython.utils.dir2 import dir2
import pytest
def test_misbehaving_object_without_trait_names():

    class MisbehavingGetattr:

        def __getattr__(self, attr):
            raise KeyError('I should be caught')

        def some_method(self):
            return True

    class SillierWithDir(MisbehavingGetattr):

        def __dir__(self):
            return ['some_method']
    for bad_klass in (MisbehavingGetattr, SillierWithDir):
        obj = bad_klass()
        assert obj.some_method()
        with pytest.raises(KeyError):
            obj.other_method()
        res = dir2(obj)
        assert 'some_method' in res