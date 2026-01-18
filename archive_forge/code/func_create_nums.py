import io
import math
import os
import typing
import weakref
def create_nums(labels):
    """Return concatenated string of all labels rules.

        Args:
            labels: (list) dictionaries as created by function 'rule_dict'.
        Returns:
            PDF compatible string for page label definitions, ready to be
            enclosed in PDF array 'Nums[...]'.
        """
    labels.sort(key=lambda x: x['startpage'])
    s = ''.join([create_label_str(label) for label in labels])
    return s