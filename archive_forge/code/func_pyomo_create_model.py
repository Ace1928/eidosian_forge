from . import pmedian
def pyomo_create_model(**kwds):
    print('CREATING MODEL %s' % sorted(list(kwds.keys())))
    return pmedian.model