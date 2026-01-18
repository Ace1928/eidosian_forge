from ansible.module_utils.six import integer_types
from ansible.module_utils.six import string_types
def scrub_none_parameters(parameters, descend_into_lists=True):
    """
    Iterate over a dictionary removing any keys that have a None value

    Reference: https://github.com/ansible-collections/community.aws/issues/251
    Credit: https://medium.com/better-programming/how-to-remove-null-none-values-from-a-dictionary-in-python-1bedf1aab5e4

    :param descend_into_lists: whether or not to descend in to lists to continue to remove None values
    :param parameters: parameter dict
    :return: parameter dict with all keys = None removed
    """
    clean_parameters = {}
    for k, v in parameters.items():
        if isinstance(v, dict):
            clean_parameters[k] = scrub_none_parameters(v, descend_into_lists=descend_into_lists)
        elif descend_into_lists and isinstance(v, list):
            clean_parameters[k] = [scrub_none_parameters(vv, descend_into_lists=descend_into_lists) if isinstance(vv, dict) else vv for vv in v]
        elif v is not None:
            clean_parameters[k] = v
    return clean_parameters