import unittest
import cupy.testing._parameterized
def parameterize(*params, _ids=True):
    check_available('parameterize')
    if _ids:
        param_name = cupy.testing._parameterized._make_class_name
    else:

        def param_name(_, i, param):
            return str(i)
    params = [pytest.param(param, id=param_name('', i, param)) for i, param in enumerate(params)]

    def f(cls):
        assert not issubclass(cls, unittest.TestCase)
        if issubclass(cls, _TestingParameterizeMixin):
            raise RuntimeError('do not `@testing.parameterize` twice')
        module_name = cls.__module__
        cls = type(cls.__name__, (_TestingParameterizeMixin, cls), {})
        cls.__module__ = module_name
        cls = pytest.mark.parametrize('_cupy_testing_param', params)(cls)
        return cls
    return f