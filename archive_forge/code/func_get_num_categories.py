from enum import Enum, Flag
@classmethod
def get_num_categories(cls) -> int:
    """:returns: The number of likelihood categories in the enum."""
    return 4