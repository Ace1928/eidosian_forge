import enum
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.util.tf_export import tf_export
@classmethod
def all_but(cls, exclude):
    """Returns a tuple that enables all but the excluded options."""
    if not isinstance(exclude, (list, tuple, set)):
        exclude = (exclude,)
    return tuple(set(cls.all()) - set(exclude) - {cls.ALL})