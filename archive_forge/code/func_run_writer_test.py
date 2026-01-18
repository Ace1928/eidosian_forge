import sys
from io import StringIO
from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output
from pyomo.repn.tests.lp_diff import lp_diff
def run_writer_test():
    with LoggingIntercept() as LOG, capture_output(capture_fd=True) as OUT:
        from pyomo.opt import WriterFactory
        info = []
        for writer in sorted(WriterFactory):
            info.append('  %s: %s' % (writer, WriterFactory.doc(writer)))
            _check_log_and_out(LOG, OUT, 10, writer)
    print('Pyomo Problem Writers')
    print('---------------------')
    print('\n'.join(info))
    with LoggingIntercept() as LOG, capture_output(capture_fd=True) as OUT:
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.c = pyo.Constraint(expr=m.x >= 1)
        m.o = pyo.Objective(expr=m.x ** 2)
        from pyomo.common.tempfiles import TempfileManager
        with TempfileManager:
            fname = TempfileManager.create_tempfile(suffix='pyomo.lp_v1')
            m.write(fname, format='lp_v1')
            with open(fname, 'r') as FILE:
                data = FILE.read()
    base, test = lp_diff(_baseline, data)
    if base != test:
        print('Result did not match baseline.\nRESULT:\n%s\nBASELINE:\n%s' % (data, _baseline))
        print(data.strip().splitlines())
        print(_baseline.strip().splitlines())
        sys.exit(2)
    _check_log_and_out(LOG, OUT, 10)