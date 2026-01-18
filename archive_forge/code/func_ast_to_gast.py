from gast.astn import AstToGAst, GAstToAst
import gast
import ast
import sys
def ast_to_gast(node):
    return Ast3ToGAst().visit(node)