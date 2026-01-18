import re
def parse_aws_arn(arn):
    """
    Based on https://docs.aws.amazon.com/IAM/latest/UserGuide/reference-arns.html

    The following are the general formats for ARNs.
        arn:partition:service:region:account-id:resource-id
        arn:partition:service:region:account-id:resource-type/resource-id
        arn:partition:service:region:account-id:resource-type:resource-id
    The specific formats depend on the resource.
    The ARNs for some resources omit the Region, the account ID, or both the Region and the account ID.

    Note: resource_type handling is very naive, for complex cases it may be necessary to use
    "resource" directly instead of resource_type, this will include the resource type and full ID,
    including all paths.
    """
    m = re.search('arn:(aws(-([a-z\\-]+))?):([\\w-]+):([a-z0-9\\-]*):(\\d*|aws|aws-managed):(.*)', arn)
    if m is None:
        return None
    result = dict()
    result.update(dict(partition=m.group(1)))
    result.update(dict(service=m.group(4)))
    result.update(dict(region=m.group(5)))
    result.update(dict(account_id=m.group(6)))
    result.update(dict(resource=m.group(7)))
    m2 = re.search('^(.*?)[:/](.+)$', m.group(7))
    if m2 is None:
        result.update(dict(resource_type=None, resource_id=m.group(7)))
    else:
        result.update(dict(resource_type=m2.group(1), resource_id=m2.group(2)))
    return result