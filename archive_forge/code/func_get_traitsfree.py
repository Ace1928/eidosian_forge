import os
from inspect import isclass
from copy import deepcopy
from warnings import warn
from packaging.version import Version
from traits.trait_errors import TraitError
from traits.trait_handlers import TraitDictObject, TraitListObject
from ...utils.filemanip import md5, hash_infile, hash_timestamp
from .traits_extension import (
from ... import config, __version__
def get_traitsfree(self, **kwargs):
    """Returns traited class as a dict

        Augments the trait get function to return a dictionary without
        any traits. The dictionary does not contain any attributes that
        were Undefined
        """
    out = super(BaseTraitedSpec, self).trait_get(**kwargs)
    out = self._clean_container(out, skipundefined=True)
    return out