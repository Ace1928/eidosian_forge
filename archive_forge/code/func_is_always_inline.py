from abc import ABCMeta, abstractmethod
@property
def is_always_inline(self):
    """
        True if always inline
        """
    return self._inline == 'always'