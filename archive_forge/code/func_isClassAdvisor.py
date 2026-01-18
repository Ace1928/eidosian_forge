from types import FunctionType
import sys
def isClassAdvisor(ob):
    """True if 'ob' is a class advisor function"""
    return isinstance(ob, FunctionType) and hasattr(ob, 'previousMetaclass')