from sympy.core.basic import Basic
from sympy.core.numbers import Rational
from sympy.core.singleton import S, Singleton
def test_names_in_namespace():
    d = {}
    exec('from sympy import *', d)
    for name in dir(S) + list(S._classes_to_install):
        if name.startswith('_'):
            continue
        if name == 'register':
            continue
        if isinstance(getattr(S, name), Rational):
            continue
        if getattr(S, name).__module__.startswith('sympy.physics'):
            continue
        if name in ['MySingleton', 'MySingleton_sub', 'TestSingleton']:
            continue
        if name == 'NegativeInfinity':
            continue
        assert any((getattr(S, name) is i for i in d.values())), name