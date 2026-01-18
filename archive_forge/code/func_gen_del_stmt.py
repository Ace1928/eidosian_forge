import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def gen_del_stmt(self, name_to_del):
    return ast.Delete(targets=[ast.Name(name_to_del, ast.Del())])