from CreateFps import GetMolFingerprint
from rdkit import Chem, DataStructs
from rdkit.ML.KNN.KNNRegressionModel import KNNRegressionModel
from rdkit.RDLogger import logger
import sys
import copy
import types
from optparse import Option, OptionParser, OptionValueError
def check_floatlist(option, opt, value):
    try:
        v = eval(value)
        if type(v) not in (types.ListType, types.TupleType):
            raise ValueError
        v = [float(x) for x in v]
    except ValueError:
        raise OptionValueError('option %s : invalid float list value: %r' % (opt, value))
    return v