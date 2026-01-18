from abc import ABCMeta, abstractmethod
@property
def has_cost_model(self):
    """
        True if a cost model is provided
        """
    return not (self.is_always_inline or self.is_never_inline)