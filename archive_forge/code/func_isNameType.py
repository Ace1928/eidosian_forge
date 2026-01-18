import copy
import os
import pickle
import warnings
import numpy as np
@staticmethod
def isNameType(var):
    return any((isinstance(var, t) for t in MetaArray.nameTypes))