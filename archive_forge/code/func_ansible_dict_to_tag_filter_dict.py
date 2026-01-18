from ansible.module_utils._text import to_native
from ansible.module_utils._text import to_text
from ansible.module_utils.six import string_types
def ansible_dict_to_tag_filter_dict(tags_dict):
    """Prepends "tag:" to all of the keys (not the values) in a dict
    This is useful when you're then going to build a filter including the tags.

    Note: booleans are converted to their Capitalized text form ("True" and "False"), this is
    different to ansible_dict_to_boto3_filter_list because historically we've used "to_text()" and
    AWS stores tags as strings, whereas for things which are actually booleans in AWS are returned
    as lowercase strings in filters.

    Args:
        tags_dict (dict): Dict representing AWS resource tags.

    Basic Usage:
        >>> filters = ansible_dict_to_boto3_filter_list(ansible_dict_to_tag_filter_dict(tags))

    Returns:
        Dict: A dictionary suitable for passing to ansible_dict_to_boto3_filter_list which can
        also be combined with other common filter parameters.
    """
    if not tags_dict:
        return {}
    return {_tag_name_to_filter_key(k): to_native(v) for k, v in tags_dict.items()}