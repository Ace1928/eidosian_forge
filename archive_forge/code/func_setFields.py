from collections import OrderedDict
import numpy as np
from .. import functions as fn
from .. import parametertree as ptree
from ..Qt import QtCore
def setFields(self, fields):
    """
        Set the list of fields to be used by the mapper. 
        
        The format of *fields* is::
        
            [ (fieldName, {options}), ... ]
        
        ============== ============================================================
        Field Options:
        mode           Either 'range' or 'enum' (default is range). For 'range', 
                       The user may specify a gradient of colors to be applied 
                       linearly across a specific range of values. For 'enum', 
                       the user specifies a single color for each unique value
                       (see *values* option).
        units          String indicating the units of the data for this field.
        values         List of unique values for which the user may assign a 
                       color when mode=='enum'. Optionally may specify a dict 
                       instead {value: name}.
        defaults       Dict of default values to apply to color map items when
                       they are created. Valid keys are 'colormap' to provide
                       a default color map, or otherwise they a string or tuple
                       indicating the parameter to be set, such as 'Operation' or
                       ('Channels..', 'Red').
        ============== ============================================================
        """
    self.fields = OrderedDict(fields)
    names = self.fieldNames()
    self.setAddList(names)