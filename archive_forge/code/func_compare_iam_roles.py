from collections import namedtuple
from time import sleep
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from .retries import AWSRetry
from .tagging import ansible_dict_to_boto3_tag_list
from .tagging import boto3_tag_list_to_ansible_dict
from .tagging import compare_aws_tags
from .waiters import get_waiter
def compare_iam_roles(existing_roles, target_roles, purge_roles):
    """
    Returns differences between target and existing IAM roles

        Parameters:
            existing_roles (list): Existing IAM roles
            target_roles (list): Target IAM roles
            purge_roles (bool): Remove roles not in target_roles if True

        Returns:
            roles_to_add (list): List of IAM roles to add
            roles_to_delete (list): List of IAM roles to delete
    """
    existing_roles = [dict(((k, v) for k, v in role.items() if k != 'status')) for role in existing_roles]
    roles_to_add = [role for role in target_roles if role not in existing_roles]
    roles_to_remove = [role for role in existing_roles if role not in target_roles] if purge_roles else []
    return (roles_to_add, roles_to_remove)