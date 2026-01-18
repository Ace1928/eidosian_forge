import dill
import abc
from abc import ABC
import warnings
from types import FunctionType
def test_meta_local_no_cache():
    """
    Test calling metaclass and cache registration
    """
    LocalMetaABC = abc.ABCMeta('LocalMetaABC', (), {})

    class ClassyClass:
        pass

    class KlassyClass:
        pass
    LocalMetaABC.register(ClassyClass)
    assert not issubclass(KlassyClass, LocalMetaABC)
    assert issubclass(ClassyClass, LocalMetaABC)
    res = dill.dumps((LocalMetaABC, ClassyClass, KlassyClass))
    lmabc, cc, kc = dill.loads(res)
    assert type(lmabc) == type(LocalMetaABC)
    assert not issubclass(kc, lmabc)
    assert issubclass(cc, lmabc)