import os.path as op
import inspect
import numpy as np
from ... import logging
from ..base import (
def create_interface_specs(class_name, params=None, BaseClass=TraitedSpec):
    """Create IN/Out interface specifications dynamically.

    Parameters
    ----------
    class_name: str
        The future class name(e.g, (MyClassInSpec))
    params: list of tuple
        dipy argument list
    BaseClass: TraitedSpec object
        parent class

    Returns
    -------
    newclass: object
        new nipype interface specification class

    """
    attr = {}
    if params is not None:
        for p in params:
            name, dipy_type, desc = (p[0], p[1], p[2])
            is_file = bool('files' in name or 'out_' in name)
            traits_type, is_mandatory = convert_to_traits_type(dipy_type, is_file)
            if BaseClass.__name__ == BaseInterfaceInputSpec.__name__:
                if len(p) > 3:
                    attr[name] = traits_type(p[3], desc=desc[-1], usedefault=True, mandatory=is_mandatory)
                else:
                    attr[name] = traits_type(desc=desc[-1], mandatory=is_mandatory)
            else:
                attr[name] = traits_type(p[3], desc=desc[-1], exists=True, usedefault=True)
    newclass = type(str(class_name), (BaseClass,), attr)
    return newclass