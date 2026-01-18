from inspect import getdoc
from .group import GroupMixin
from .adjoint import AdjointMixin
from .linear import LinearMixin
from .multiply import MultiplyMixin
from .tolerances import TolerancesMixin
def generate_apidocs(cls):
    """Decorator to format API docstrings for classes using Mixins.

    This runs string replacement on the docstrings of the mixin
    methods to replace the placeholder CLASS with the class
    name `cls.__name__`.

    Args:
        cls (type): The class to format docstrings.

    Returns:
        cls: the original class with updated docstrings.
    """

    def _replace_name(mixin, methods):
        if issubclass(cls, mixin):
            for i in methods:
                meth = getattr(cls, i)
                doc = getdoc(meth)
                if doc is not None:
                    meth.__doc__ = doc.replace('CLASS', cls.__name__)
    _replace_name(GroupMixin, ('tensor', 'expand', 'compose', 'dot', 'power'))
    _replace_name(AdjointMixin, ('transpose', 'conjugate', 'adjoint'))
    _replace_name(MultiplyMixin, ('_multiply',))
    _replace_name(LinearMixin, ('_add',))
    return cls