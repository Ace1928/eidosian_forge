import dill
def test_mixins():
    assert double_add(1, 2, 3) == 2 * fx
    double_add.invert()
    assert double_add(1, 2, 3) == -2 * fx
    _d = dill.copy(double_add)
    assert _d(1, 2, 3) == -2 * fx
    assert _d.__wrapped__(1, 2, 3) == fx
    if not double_add.inverted[0]:
        double_add.invert()
    ds = dill.source
    dd = dill.detect
    assert ds.getsource(dd.freevars(quadish)['f']) == '@quad_factory(a=0,b=4,c=0)\ndef quadish(x):\n  return x+1\n'
    assert ds.getsource(dd.freevars(quadruple)['f']) == '@doubler\ndef quadruple(x):\n  return 2*x\n'
    assert ds.importable(quadish, source=False) == 'from %s import quadish\n' % __name__
    assert ds.importable(quadruple, source=False) == 'from %s import quadruple\n' % __name__
    assert ds.importable(quadratic, source=False) == 'from %s import quadratic\n' % __name__
    assert ds.importable(double_add, source=False) == 'from %s import double_add\n' % __name__
    assert ds.importable(quadruple, source=True) == 'def doubler(f):\n  def inner(*args, **kwds):\n    fx = f(*args, **kwds)\n    return 2*fx\n  return inner\n\n@doubler\ndef quadruple(x):\n  return 2*x\n'
    result = ds.importable(quadish, source=True)
    a, b, c, _, result = result.split('\n', 4)
    assert result == 'def quad_factory(a=1,b=1,c=0):\n  def dec(f):\n    def func(*args,**kwds):\n      fx = f(*args,**kwds)\n      return a*fx**2 + b*fx + c\n    return func\n  return dec\n\n@quad_factory(a=0,b=4,c=0)\ndef quadish(x):\n  return x+1\n'
    assert set([a, b, c]) == set(['a = 0', 'c = 0', 'b = 4'])
    result = ds.importable(quadratic, source=True)
    a, b, c, result = result.split('\n', 3)
    assert result == '\ndef dec(f):\n  def func(*args,**kwds):\n    fx = f(*args,**kwds)\n    return a*fx**2 + b*fx + c\n  return func\n'
    assert set([a, b, c]) == set(['a = 1', 'c = 0', 'b = 1'])
    result = ds.importable(double_add, source=True)
    a, b, c, d, _, result = result.split('\n', 5)
    assert result == 'def quad(a=1, b=1, c=0):\n  inverted = [False]\n  def invert():\n    inverted[0] = not inverted[0]\n  def dec(f):\n    def func(*args, **kwds):\n      x = f(*args, **kwds)\n      if inverted[0]: x = -x\n      return a*x**2 + b*x + c\n    func.__wrapped__ = f\n    func.invert = invert\n    func.inverted = inverted\n    return func\n  return dec\n\n@quad(a=0,b=2)\ndef double_add(*args):\n  return sum(args)\n'
    assert set([a, b, c, d]) == set(['a = 0', 'c = 0', 'b = 2', 'inverted = [True]'])