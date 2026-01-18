import importlib
import pkgutil
import re
import sys
import pbr.version
def check_traits(traits, prefix=None):
    """Returns a tuple of two trait string sets, the first set contains valid
    traits, and the second contains others.

    :param traits: An iterable contains trait strings.
    :param prefix: Optional string prefix to filter by. e.g. 'HW_'
    """
    trait_set = set(traits)
    valid_trait_set = set(get_traits(prefix))
    valid_traits = trait_set & valid_trait_set
    return (valid_traits, trait_set - valid_traits)