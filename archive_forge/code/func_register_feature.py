import contextlib
import sys
from typing import TYPE_CHECKING, Set, cast
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from .. import config, controldir, errors, lockdir
from .. import transport as _mod_transport
from ..trace import mutter, note, warning
from ..transport import do_catching_redirections, local
@classmethod
def register_feature(cls, name):
    """Register a feature as being present.

        :param name: Name of the feature
        """
    if b' ' in name:
        raise ValueError('spaces are not allowed in feature names')
    if name in cls._present_features:
        raise FeatureAlreadyRegistered(name)
    cls._present_features.add(name)