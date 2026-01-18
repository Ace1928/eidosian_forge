from collections import namedtuple
from collections.abc import Mapping
import copy
import inspect
import re
import textwrap
from statsmodels.tools.sm_exceptions import ParseError
def insert_parameters(self, after, parameters):
    """
        Parameters
        ----------
        after : {None, str}
            If None, inset the parameters before the first parameter in the
            docstring.
        parameters : Parameter, list[Parameter]
            A Parameter of a list of Parameters.
        """
    if self._docstring is None:
        return
    if isinstance(parameters, Parameter):
        parameters = [parameters]
    if after is None:
        self._ds['Parameters'] = parameters + self._ds['Parameters']
    else:
        loc = -1
        for i, param in enumerate(self._ds['Parameters']):
            if param.name == after:
                loc = i + 1
                break
        if loc < 0:
            raise ValueError()
        params = self._ds['Parameters'][:loc] + parameters
        params += self._ds['Parameters'][loc:]
        self._ds['Parameters'] = params