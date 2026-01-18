from __future__ import absolute_import, division, print_function
def helper_to_numeric_value(elements, value):
    """Converts string values to integers

    Parameters:
        elements: list of elements to enumerate
        value: string value

    Returns:
        int: converted integer
    """
    if value is None:
        return None
    for index, element in enumerate(elements):
        if isinstance(element, str) and element.lower() == value.lower():
            return index
        if isinstance(element, list):
            for deep_element in element:
                if isinstance(deep_element, str) and deep_element.lower() == value.lower():
                    return index