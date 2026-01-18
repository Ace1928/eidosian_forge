from pythran.passmanager import Transformation
from pythran.tables import MODULES, pythran_ward
from pythran.syntax import PythranSyntaxError
import gast as ast
import logging
import os
def rename_top_level_functions(self, node):
    for stmt in node.body:
        if isinstance(stmt, ast.FunctionDef):
            self.rename(stmt, 'name')
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    self.rename(target, 'id')