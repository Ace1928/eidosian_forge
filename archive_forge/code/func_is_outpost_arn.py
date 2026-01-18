import re
def is_outpost_arn(arn):
    """
    Validates that the ARN is for an AWS Outpost


    API Specification Document:
    https://docs.aws.amazon.com/outposts/latest/APIReference/API_Outpost.html
    """
    details = parse_aws_arn(arn)
    if not details:
        return False
    service = details.get('service') or ''
    if service.lower() != 'outposts':
        return False
    resource = details.get('resource') or ''
    if not re.match('^outpost/op-[a-f0-9]{17}$', resource):
        return False
    return True