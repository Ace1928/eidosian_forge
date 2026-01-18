import abc
from tensorflow.python.util.tf_export import tf_export
@abc.abstractproperty
def parents(self):
    """Returns a list of immediate raw feature and FeatureColumn dependencies.

    For example:
    # For the following feature columns
    a = numeric_column('f1')
    c = crossed_column(a, 'f2')
    # The expected parents are:
    a.parents = ['f1']
    c.parents = [a, 'f2']
    """
    pass