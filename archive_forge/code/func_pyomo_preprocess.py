from . import pmedian
def pyomo_preprocess(**kwds):
    print('PREPROCESSING %s' % sorted(list(kwds.keys())))