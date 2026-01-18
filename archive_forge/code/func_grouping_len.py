from dash.exceptions import InvalidCallbackReturnValue
from ._utils import AttributeDict, stringify_id
def grouping_len(grouping):
    """
    Get the length of a grouping. The length equal to the number of scalar values
    contained in the grouping, which is equivalent to the length of the list that would
    result from calling flatten_grouping on the grouping value.

    :param grouping: The grouping value to calculate the length of
    :return: non-negative integer
    """
    if isinstance(grouping, (tuple, list)):
        return sum([grouping_len(group_el) for group_el in grouping])
    if isinstance(grouping, dict):
        return sum([grouping_len(group_el) for group_el in grouping.values()])
    return 1