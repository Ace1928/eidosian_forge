import enum
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.util.tf_export import tf_export
def as_tuple(self):
    return (self.recursive, self.user_requested, self.internal_convert_user_code, self.optional_features)