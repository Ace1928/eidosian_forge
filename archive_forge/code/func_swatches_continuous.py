from .colorbrewer import (  # noqa: F401
from .cmocean import (  # noqa: F401
from .carto import (  # noqa: F401
from .plotlyjs import Picnic, Portland, Picnic_r, Portland_r  # noqa: F401
from ._swatches import _swatches, _swatches_continuous
def swatches_continuous(template=None):
    return _swatches_continuous(__name__, globals(), template)