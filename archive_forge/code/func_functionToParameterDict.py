import contextlib
import functools
import inspect
import pydoc
from .. import functions as fn
from . import Parameter
from .parameterTypes import ActionGroupParameter
def functionToParameterDict(self, function, **overrides):
    """
        Converts a function into a list of child parameter dicts
        """
    children = []
    name = function.__name__
    btnOpts = dict(**self._makePopulatedActionTemplate(name), visible=False)
    out = dict(name=name, type='_actiongroup', children=children, button=btnOpts)
    if self.titleFormat is not None:
        out['title'] = self._nameToTitle(name, forwardStringTitle=True)
    funcParams = inspect.signature(function).parameters
    if function.__doc__:
        synopsis, _ = pydoc.splitdoc(function.__doc__)
        if synopsis:
            out.setdefault('tip', synopsis)
            out['button'].setdefault('tip', synopsis)
    checkNames = list(funcParams)
    parameterKinds = [p.kind for p in funcParams.values()]
    _positional = inspect.Parameter.VAR_POSITIONAL
    _keyword = inspect.Parameter.VAR_KEYWORD
    if _keyword in parameterKinds:
        del checkNames[-1]
        notInSignature = [n for n in overrides if n not in checkNames]
        checkNames.extend(notInSignature)
    if _positional in parameterKinds:
        del checkNames[parameterKinds.index(_positional)]
    for name in checkNames:
        param = funcParams.get(name)
        pgDict = self.createFunctionParameter(name, param, overrides.get(name, {}))
        children.append(pgDict)
    return out