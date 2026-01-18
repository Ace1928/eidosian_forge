import __future__
import builtins
import ast
import collections
import contextlib
import doctest
import functools
import os
import re
import string
import sys
import warnings
from pyflakes import messages
def handleNodeLoad(self, node, parent):
    name = getNodeName(node)
    if not name:
        return
    can_access_class_vars = None
    importStarred = None
    for scope in self.scopeStack[-1::-1]:
        if isinstance(scope, ClassScope):
            if name == '__class__':
                return
            elif can_access_class_vars is False:
                continue
        binding = scope.get(name, None)
        if isinstance(binding, Annotation) and (not self._in_postponed_annotation):
            scope[name].used = (self.scope, node)
            continue
        if name == 'print' and isinstance(binding, Builtin):
            if isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.RShift):
                self.report(messages.InvalidPrintSyntax, node)
        try:
            scope[name].used = (self.scope, node)
            n = scope[name]
            if isinstance(n, Importation) and n._has_alias():
                try:
                    scope[n.fullName].used = (self.scope, node)
                except KeyError:
                    pass
        except KeyError:
            pass
        else:
            return
        importStarred = importStarred or scope.importStarred
        if can_access_class_vars is not False:
            can_access_class_vars = isinstance(scope, (TypeScope, GeneratorScope))
    if importStarred:
        from_list = []
        for scope in self.scopeStack[-1::-1]:
            for binding in scope.values():
                if isinstance(binding, StarImportation):
                    binding.used = (self.scope, node)
                    from_list.append(binding.fullName)
        from_list = ', '.join(sorted(from_list))
        self.report(messages.ImportStarUsage, node, name, from_list)
        return
    if name == '__path__' and os.path.basename(self.filename) == '__init__.py':
        return
    if name in DetectClassScopedMagic.names and isinstance(self.scope, ClassScope):
        return
    if 'NameError' not in self.exceptHandlers[-1]:
        self.report(messages.UndefinedName, node, name)