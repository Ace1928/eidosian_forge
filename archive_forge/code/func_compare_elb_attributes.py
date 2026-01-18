import traceback
from copy import deepcopy
from .ec2 import get_ec2_security_group_ids_from_names
from .elb_utils import convert_tg_name_to_arn
from .elb_utils import get_elb
from .elb_utils import get_elb_listener
from .retries import AWSRetry
from .tagging import ansible_dict_to_boto3_tag_list
from .tagging import boto3_tag_list_to_ansible_dict
from .waiters import get_waiter
def compare_elb_attributes(self):
    """
        Compare user attributes with current ELB attributes
        :return: bool True if they match otherwise False
        """
    update_attributes = []
    if self.access_logs_enabled is not None and str(self.access_logs_enabled).lower() != self.elb_attributes['access_logs_s3_enabled']:
        update_attributes.append({'Key': 'access_logs.s3.enabled', 'Value': str(self.access_logs_enabled).lower()})
    if self.access_logs_s3_bucket is not None and self.access_logs_s3_bucket != self.elb_attributes['access_logs_s3_bucket']:
        update_attributes.append({'Key': 'access_logs.s3.bucket', 'Value': self.access_logs_s3_bucket})
    if self.access_logs_s3_prefix is not None and self.access_logs_s3_prefix != self.elb_attributes['access_logs_s3_prefix']:
        update_attributes.append({'Key': 'access_logs.s3.prefix', 'Value': self.access_logs_s3_prefix})
    if self.deletion_protection is not None and str(self.deletion_protection).lower() != self.elb_attributes['deletion_protection_enabled']:
        update_attributes.append({'Key': 'deletion_protection.enabled', 'Value': str(self.deletion_protection).lower()})
    if self.idle_timeout is not None and str(self.idle_timeout) != self.elb_attributes['idle_timeout_timeout_seconds']:
        update_attributes.append({'Key': 'idle_timeout.timeout_seconds', 'Value': str(self.idle_timeout)})
    if self.http2 is not None and str(self.http2).lower() != self.elb_attributes['routing_http2_enabled']:
        update_attributes.append({'Key': 'routing.http2.enabled', 'Value': str(self.http2).lower()})
    if self.http_desync_mitigation_mode is not None and str(self.http_desync_mitigation_mode).lower() != self.elb_attributes['routing_http_desync_mitigation_mode']:
        update_attributes.append({'Key': 'routing.http.desync_mitigation_mode', 'Value': str(self.http_desync_mitigation_mode).lower()})
    if self.http_drop_invalid_header_fields is not None and str(self.http_drop_invalid_header_fields).lower() != self.elb_attributes['routing_http_drop_invalid_header_fields_enabled']:
        update_attributes.append({'Key': 'routing.http.drop_invalid_header_fields.enabled', 'Value': str(self.http_drop_invalid_header_fields).lower()})
    if self.http_x_amzn_tls_version_and_cipher_suite is not None and str(self.http_x_amzn_tls_version_and_cipher_suite).lower() != self.elb_attributes['routing_http_x_amzn_tls_version_and_cipher_suite_enabled']:
        update_attributes.append({'Key': 'routing.http.x_amzn_tls_version_and_cipher_suite.enabled', 'Value': str(self.http_x_amzn_tls_version_and_cipher_suite).lower()})
    if self.http_xff_client_port is not None and str(self.http_xff_client_port).lower() != self.elb_attributes['routing_http_xff_client_port_enabled']:
        update_attributes.append({'Key': 'routing.http.xff_client_port.enabled', 'Value': str(self.http_xff_client_port).lower()})
    if self.waf_fail_open is not None and str(self.waf_fail_open).lower() != self.elb_attributes['waf_fail_open_enabled']:
        update_attributes.append({'Key': 'waf.fail_open.enabled', 'Value': str(self.waf_fail_open).lower()})
    if update_attributes:
        return False
    else:
        return True