from boto.s3 import user
from boto.s3 import key
from boto import handler
import xml.sax
def cancel_upload(self):
    """
        Cancels a MultiPart Upload operation.  The storage consumed by
        any previously uploaded parts will be freed. However, if any
        part uploads are currently in progress, those part uploads
        might or might not succeed. As a result, it might be necessary
        to abort a given multipart upload multiple times in order to
        completely free all storage consumed by all parts.
        """
    self.bucket.cancel_multipart_upload(self.key_name, self.id)