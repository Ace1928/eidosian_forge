import contextlib
import functools
import inspect
import pydoc
from .. import functions as fn
from . import Parameter
from .parameterTypes import ActionGroupParameter
def createFunctionParameter(self, name, signatureParameter, overridesInfo):
    """
        Constructs a dict ready for insertion into a group parameter based on the
        provided information in the ``inspect.signature`` parameter, user-specified
        overrides, and true parameter name. Parameter signature information is
        considered the most "overridable", followed by documentation specifications.
        User overrides should be given the highest priority, i.e. not usurped by
        parameter default information.

        Parameters
        ----------
        name : str
            Name of the parameter, comes from function signature
        signatureParameter : inspect.Parameter
            Information from the function signature, parsed by ``inspect``
        overridesInfo : dict
            User-specified overrides for this parameter. Can be a dict of options
            accepted by :class:`~pyqtgraph.parametertree.Parameter` or a value
        """
    if signatureParameter is not None and signatureParameter.default is not signatureParameter.empty:
        default = signatureParameter.default
        signatureDict = {'value': default, 'type': type(default).__name__}
    else:
        signatureDict = {}
    pgDict = signatureDict.copy()
    if not isinstance(overridesInfo, dict):
        overridesInfo = {'value': overridesInfo}
    pgDict.update(overridesInfo)
    pgDict['name'] = name
    pgDict.setdefault('value', PARAM_UNSET)
    if self.titleFormat is not None:
        pgDict.setdefault('title', self._nameToTitle(name))
    pgDict.setdefault('type', type(pgDict['value']).__name__)
    return pgDict