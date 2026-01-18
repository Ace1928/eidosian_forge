from reportlab.lib.validators import isAnything, DerivedValue
from reportlab.lib.utils import isSeq
from reportlab import rl_config
def hook__setattr__(obj):
    if not hasattr(obj, '__attrproxy__'):
        C = obj.__class__
        import new
        obj.__class__ = new.classobj(C.__name__, (C,) + C.__bases__, {'__attrproxy__': [], '__setattr__': lambda self, k, v, osa=getattr(obj, '__setattr__', None), hook=hook: hook(self, k, v, osa)})