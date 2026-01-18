from . import pmedian
def pyomo_print_instance(**kwds):
    print('PRINTING INSTANCE %s' % sorted(list(kwds.keys())))