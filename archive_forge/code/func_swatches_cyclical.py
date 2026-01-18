from ._swatches import _swatches, _swatches_continuous, _swatches_cyclical
def swatches_cyclical(template=None):
    return _swatches_cyclical(__name__, globals(), template)