from boto.s3 import user
from boto.s3 import key
from boto import handler
import xml.sax
def part_lister(mpupload, part_number_marker=None):
    """
    A generator function for listing parts of a multipart upload.
    """
    more_results = True
    part = None
    while more_results:
        parts = mpupload.get_all_parts(None, part_number_marker)
        for part in parts:
            yield part
        part_number_marker = mpupload.next_part_number_marker
        more_results = mpupload.is_truncated