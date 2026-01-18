from collections import OrderedDict
import numpy as np
from .. import functions as fn
from .. import parametertree as ptree
from ..Qt import QtCore
def filterData(self, data):
    if len(data) == 0:
        return data
    return data[self.generateMask(data)]