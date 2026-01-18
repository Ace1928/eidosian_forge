from sympy.interactive.session import (init_ipython_session,
from sympy.core import Symbol, Rational, Integer
from sympy.external import import_module
from sympy.testing.pytest import raises
def test_ipythonprinting():
    app = init_ipython_session()
    app.run_cell('ip = get_ipython()')
    app.run_cell('inst = ip.instance()')
    app.run_cell('format = inst.display_formatter.format')
    app.run_cell('from sympy import Symbol')
    app.run_cell("a = format(Symbol('pi'))")
    app.run_cell("a2 = format(Symbol('pi')**2)")
    if int(ipython.__version__.split('.')[0]) < 1:
        assert app.user_ns['a']['text/plain'] == 'pi'
        assert app.user_ns['a2']['text/plain'] == 'pi**2'
    else:
        assert app.user_ns['a'][0]['text/plain'] == 'pi'
        assert app.user_ns['a2'][0]['text/plain'] == 'pi**2'
    app.run_cell('from sympy import init_printing')
    app.run_cell('init_printing()')
    app.run_cell("a = format(Symbol('pi'))")
    app.run_cell("a2 = format(Symbol('pi')**2)")
    if int(ipython.__version__.split('.')[0]) < 1:
        assert app.user_ns['a']['text/plain'] in ('π', 'pi')
        assert app.user_ns['a2']['text/plain'] in (' 2\nπ ', '  2\npi ')
    else:
        assert app.user_ns['a'][0]['text/plain'] in ('π', 'pi')
        assert app.user_ns['a2'][0]['text/plain'] in (' 2\nπ ', '  2\npi ')