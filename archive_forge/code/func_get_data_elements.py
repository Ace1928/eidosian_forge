import re
import numpy as np
from xml.dom import minidom
def get_data_elements(name, dtype):
    """
    return the right type of the element value
    """
    if dtype is int:
        data = str2int(getNodeText(name))
        if len(data) > 1:
            return np.array(data)
        elif len(data) == 1:
            return data[0]
        else:
            raise ValueError('len(data)<1 ??')
    elif dtype is float:
        data = str2float(getNodeText(name))
        if len(data) > 1:
            return np.array(data)
        elif len(data) == 1:
            return data[0]
        else:
            raise ValueError('len(data)<1 ??')
    elif dtype is str:
        return getNodeText(name)
    else:
        raise ValueError('not implemented')