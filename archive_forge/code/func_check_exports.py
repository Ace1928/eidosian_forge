from pythran.tables import MODULES
from pythran.intrinsic import Class
from pythran.typing import Tuple, List, Set, Dict
from pythran.utils import isstr
from pythran import metadata
import beniget
import gast as ast
import logging
import numpy as np
def check_exports(pm, mod, specs):
    """
    Does nothing but raising PythranSyntaxError if specs
    references an undefined global
    """
    from pythran.analyses.argument_effects import ArgumentEffects
    mod_functions = {node.name: node for node in mod.body if isinstance(node, ast.FunctionDef)}
    argument_effects = pm.gather(ArgumentEffects, mod)
    for fname, signatures in specs.functions.items():
        try:
            fnode = mod_functions[fname]
        except KeyError:
            raise PythranSyntaxError('Invalid spec: exporting undefined function `{}`'.format(fname))
        is_global = metadata.get(fnode.body[0], metadata.StaticReturn)
        if is_global and signatures:
            raise PythranSyntaxError('Invalid spec: exporting global `{}` as a function'.format(fname))
        if not is_global and (not signatures):
            raise PythranSyntaxError('Invalid spec: exporting function `{}` as a global'.format(fname))
        ae = argument_effects[fnode]
        for signature in signatures:
            args_count = len(fnode.args.args)
            if len(signature) > args_count:
                raise PythranSyntaxError('Too many arguments when exporting `{}`'.format(fname))
            elif len(signature) < args_count - len(fnode.args.defaults):
                raise PythranSyntaxError('Not enough arguments when exporting `{}`'.format(fname))
            for i, ty in enumerate(signature):
                if ae[i] and isinstance(ty, (List, Tuple, Dict, Set)):
                    logger.warning("Exporting function '{}' that modifies its {} argument. Beware that this argument won't be modified at Python call site".format(fname, ty.__class__.__qualname__))