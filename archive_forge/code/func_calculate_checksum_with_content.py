import string
from urllib.parse import urlparse
from ansible.module_utils.basic import to_text
def calculate_checksum_with_content(client, parts, bucket, obj, versionId, content):
    digests = []
    offset = 0
    for head in s3_head_objects(client, parts, bucket, obj, versionId):
        length = int(head['ContentLength'])
        digests.append(md5(content[offset:offset + length]).digest())
        offset += length
    digest_squared = b''.join(digests)
    return f'"{md5(digest_squared).hexdigest()}-{len(digests)}"'