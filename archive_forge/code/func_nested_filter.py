from traitlets import Bool, Set
from .base import Preprocessor
def nested_filter(self, items, mask):
    """Get the nested filter for items given a mask."""
    keep_current = self.current_mask(mask)
    keep_nested_lookup = self.nested_masks(mask)
    for k, v in items:
        keep_nested = keep_nested_lookup.get(k)
        if k in keep_current:
            if keep_nested is not None:
                if isinstance(v, dict):
                    yield (k, dict(self.nested_filter(v.items(), keep_nested)))
            else:
                yield (k, v)