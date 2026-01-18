import dill
import abc
from abc import ABC
import warnings
from types import FunctionType
def test_abc_local():
    """
    Test using locally scoped ABC class
    """

    class LocalABC(ABC):

        @abc.abstractmethod
        def foo(self):
            pass

        def baz(self):
            return repr(self)
    labc = dill.copy(LocalABC)
    assert labc is not LocalABC
    assert type(labc) is type(LocalABC)

    class Real(labc):

        def foo(self):
            return 'True!'

        def baz(self):
            return 'My ' + super(Real, self).baz()
    real = Real()
    assert real.foo() == 'True!'
    try:
        labc()
    except TypeError as e:
        pass
    else:
        print('Failed to raise type error')
        assert False
    labc2, pik = dill.copy((labc, Real()))
    assert 'Real' == type(pik).__name__
    assert '.Real' in type(pik).__qualname__
    assert type(pik) is not Real
    assert labc2 is not LocalABC
    assert labc2 is not labc
    assert isinstance(pik, labc2)
    assert not isinstance(pik, labc)
    assert not isinstance(pik, LocalABC)
    assert pik.baz() == 'My ' + repr(pik)