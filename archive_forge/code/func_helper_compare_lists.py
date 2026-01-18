from __future__ import absolute_import, division, print_function
def helper_compare_lists(l1, l2, diff_dict):
    """
    Compares l1 and l2 lists and adds the items that are different
    to the diff_dict dictionary.
    Used in recursion with helper_compare_dictionaries() function.

    Parameters:
        l1: first list to compare
        l2: second list to compare
        diff_dict: dictionary to store the difference

    Returns:
        dict: items that are different
    """
    if len(l1) != len(l2):
        diff_dict.append(l1)
        return diff_dict
    for i, item in enumerate(l1):
        if isinstance(item, dict):
            for item2 in l2:
                diff_dict2 = {}
                diff_dict2 = helper_compare_dictionaries(item, item2, diff_dict2)
                if len(diff_dict2) == 0:
                    break
            if len(diff_dict2) != 0:
                diff_dict.insert(i, item)
        elif item != l2[i]:
            diff_dict.append(item)
    while {} in diff_dict:
        diff_dict.remove({})
    return diff_dict