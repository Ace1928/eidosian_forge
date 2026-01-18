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
def checkDeadScopes(self):
    """
        Look at scopes which have been fully examined and report names in them
        which were imported but unused.
        """
    for scope in self.deadScopes:
        if isinstance(scope, ClassScope):
            continue
        if isinstance(scope, FunctionScope):
            for name, binding in scope.unused_assignments():
                self.report(messages.UnusedVariable, binding.source, name)
            for name, binding in scope.unused_annotations():
                self.report(messages.UnusedAnnotation, binding.source, name)
        all_binding = scope.get('__all__')
        if all_binding and (not isinstance(all_binding, ExportBinding)):
            all_binding = None
        if all_binding:
            all_names = set(all_binding.names)
            undefined = [name for name in all_binding.names if name not in scope]
        else:
            all_names = undefined = []
        if undefined:
            if not scope.importStarred and os.path.basename(self.filename) != '__init__.py':
                for name in undefined:
                    self.report(messages.UndefinedExport, scope['__all__'].source, name)
            if scope.importStarred:
                from_list = []
                for binding in scope.values():
                    if isinstance(binding, StarImportation):
                        binding.used = all_binding
                        from_list.append(binding.fullName)
                from_list = ', '.join(sorted(from_list))
                for name in undefined:
                    self.report(messages.ImportStarUsage, scope['__all__'].source, name, from_list)
        for value in scope.values():
            if isinstance(value, Importation):
                used = value.used or value.name in all_names
                if not used:
                    messg = messages.UnusedImport
                    self.report(messg, value.source, str(value))
                for node in value.redefined:
                    if isinstance(self.getParent(node), FOR_TYPES):
                        messg = messages.ImportShadowedByLoopVar
                    elif used:
                        continue
                    else:
                        messg = messages.RedefinedWhileUnused
                    self.report(messg, node, value.name, value.source)