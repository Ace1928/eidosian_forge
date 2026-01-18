import sys
from io import StringIO
from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output
from pyomo.repn.tests.lp_diff import lp_diff
def run_transformationfactory_test():
    with LoggingIntercept() as LOG, capture_output(capture_fd=True) as OUT:
        info = []
        for t in sorted(pyo.TransformationFactory):
            _doc = pyo.TransformationFactory.doc(t)
            info.append('   %s: %s' % (t, _doc))
            if 'DEPRECATED' not in _doc:
                pyo.TransformationFactory(t)
            _check_log_and_out(LOG, OUT, 30, t)
        bigm = pyo.TransformationFactory('gdp.bigm')
    print('')
    print('Pyomo Transformations')
    print('---------------------')
    print('\n'.join(info))
    if not isinstance(bigm, pyo.Transformation):
        print('TransformationFactory(gdp.bigm) did not return a transformation')
        sys.exit(4)
    _check_log_and_out(LOG, OUT, 30)