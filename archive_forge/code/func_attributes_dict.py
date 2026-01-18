from enum import IntFlag, auto
from typing import Dict, Tuple
from ._utils import deprecate_with_replacement
@classmethod
def attributes_dict(cls) -> Dict[str, str]:
    """
        Get a dictionary of attribute keys and their human-readable names.

        This method returns a dictionary where the keys are the attribute
        constants defined in the CheckboxRadioButtonAttributes class and the
        values are their corresponding human-readable names. These attributes
        correspond to the entries that are common to all field dictionaries as
        specified in the PDF 1.7 reference.

        Returns:
            A dictionary containing attribute keys and their names.
        """
    return {cls.Opt: 'Options'}