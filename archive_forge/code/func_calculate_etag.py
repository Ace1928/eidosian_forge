import string
from urllib.parse import urlparse
from ansible.module_utils.basic import to_text
def calculate_etag(module, filename, etag, s3, bucket, obj, version=None):
    if not HAS_MD5:
        return None
    if '-' in etag:
        parts = int(etag[1:-1].split('-')[1])
        try:
            return calculate_checksum_with_file(s3, parts, bucket, obj, version, filename)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            module.fail_json_aws(e, msg='Failed to get head object')
    else:
        return f'"{module.md5(filename)}"'