import base64
import numbers
import textwrap
import uuid
from importlib import import_module
import copy
import io
import re
import sys
import warnings
from _plotly_utils.optional_imports import get_module
def raise_invalid_val(self, v, inds=None):
    """
        Helper method to raise an informative exception when an invalid
        value is passed to the validate_coerce method.

        Parameters
        ----------
        v :
            Value that was input to validate_coerce and could not be coerced
        inds: list of int or None (default)
            Indexes to display after property name. e.g. if self.plotly_name
            is 'prop' and inds=[2, 1] then the name in the validation error
            message will be 'prop[2][1]`
        Raises
        -------
        ValueError
        """
    name = self.plotly_name
    if inds:
        for i in inds:
            name += '[' + str(i) + ']'
    raise ValueError("\n    Invalid value of type {typ} received for the '{name}' property of {pname}\n        Received value: {v}\n\n{valid_clr_desc}".format(name=name, pname=self.parent_name, typ=type_str(v), v=repr(v), valid_clr_desc=self.description()))