import warnings
from . import exceptions as exc
from . import misc
from . import normalizers
from . import validators
def normalized_equality(self, other_ref):
    """Compare this URIReference to another URIReference.

        :param URIReference other_ref: (required), The reference with which
            we're comparing.
        :returns: ``True`` if the references are equal, ``False`` otherwise.
        :rtype: bool
        """
    return tuple(self.normalize()) == tuple(other_ref.normalize())