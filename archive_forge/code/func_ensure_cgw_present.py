from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def ensure_cgw_present(self, bgp_asn, ip_address):
    if not bgp_asn:
        bgp_asn = 65000
    response = self.ec2.create_customer_gateway(DryRun=False, Type='ipsec.1', PublicIp=ip_address, BgpAsn=bgp_asn)
    return response