from . import pmedian
def pyomo_save_instance(**kwds):
    print('SAVE INSTANCE %s' % sorted(list(kwds.keys())))