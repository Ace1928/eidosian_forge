from ansible.module_utils._text import to_native
from ansible.module_utils._text import to_text
from ansible.module_utils.six import string_types
def boto3_tag_specifications(tags_dict, types=None):
    """Converts a list of resource types and a flat dictionary of key:value pairs representing AWS
    resource tags to a TagSpecification object.

    https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_TagSpecification.html

    Args:
        tags_dict (dict): Dict representing AWS resource tags.
        types (list) A list of resource types to be tagged.
    Basic Usage:
        >>> tags_dict = {'MyTagKey': 'MyTagValue'}
        >>> boto3_tag_specifications(tags_dict, ['instance'])
        [
            {
                'ResourceType': 'instance',
                'Tags': [
                    {
                        'Key': 'MyTagKey',
                        'Value': 'MyTagValue'
                    }
                ]
            }
        ]
    Returns:
        List: List of dictionaries representing an AWS Tag Specification
    """
    if not tags_dict:
        return None
    specifications = list()
    tag_list = ansible_dict_to_boto3_tag_list(tags_dict)
    if not types:
        specifications.append(dict(Tags=tag_list))
        return specifications
    if isinstance(types, string_types):
        types = [types]
    for type_name in types:
        specifications.append(dict(ResourceType=type_name, Tags=tag_list))
    return specifications