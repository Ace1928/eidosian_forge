import string
from urllib.parse import urlparse
from ansible.module_utils.basic import to_text
def calculate_checksum_with_file(client, parts, bucket, obj, versionId, filename):
    digests = []
    with open(filename, 'rb') as f:
        for head in s3_head_objects(client, parts, bucket, obj, versionId):
            digests.append(md5(f.read(int(head['ContentLength']))).digest())
    digest_squared = b''.join(digests)
    return f'"{md5(digest_squared).hexdigest()}-{len(digests)}"'