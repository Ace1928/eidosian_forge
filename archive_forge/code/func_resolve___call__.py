from .. import types, utils, errors
import operator
from .templates import (AttributeTemplate, ConcreteTemplate, AbstractTemplate,
from .builtins import normalize_1d_index
def resolve___call__(self, classty):
    """
        Resolve the named tuple constructor, aka the class's __call__ method.
        """
    instance_class = classty.instance_class
    pysig = utils.pysignature(instance_class)

    def typer(*args, **kws):
        try:
            bound = pysig.bind(*args, **kws)
        except TypeError as e:
            msg = "In '%s': %s" % (instance_class, e)
            e.args = (msg,)
            raise
        assert not bound.kwargs
        return types.BaseTuple.from_types(bound.args, instance_class)
    typer.pysig = pysig
    return types.Function(make_callable_template(self.key, typer))