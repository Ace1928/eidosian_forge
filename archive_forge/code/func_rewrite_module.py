from llvmlite import ir
from llvmlite.ir.transforms import Visitor, CallVisitor
def rewrite_module(mod, options):
    """
    Rewrite the given LLVM module to use fastmath everywhere.
    """
    flags = options.flags
    FastFloatBinOpVisitor(flags).visit(mod)
    FastFloatCallVisitor(flags).visit(mod)