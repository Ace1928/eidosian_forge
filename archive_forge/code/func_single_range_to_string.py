from enum import Enum
@staticmethod
def single_range_to_string(left, right):
    if left != right:
        return '{}-{}'.format(left, right)
    else:
        return '{}'.format(left)