import datetime
import fnmatch
import mimetypes
import os
import stat as osstat  # os.stat constants
from ansible.module_utils._text import to_text
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.etag import calculate_multipart_etag
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def gather_files(fileroot, include=None, exclude=None):
    ret = []
    if os.path.isfile(fileroot):
        fullpath = fileroot
        fstat = os.stat(fullpath)
        path_array = fileroot.split('/')
        chopped_path = path_array[-1]
        f_size = fstat[osstat.ST_SIZE]
        f_modified_epoch = fstat[osstat.ST_MTIME]
        ret.append({'fullpath': fullpath, 'chopped_path': chopped_path, 'modified_epoch': f_modified_epoch, 'bytes': f_size})
    else:
        for dirpath, dirnames, filenames in os.walk(fileroot):
            for fn in filenames:
                fullpath = os.path.join(dirpath, fn)
                if include:
                    found = False
                    for x in include.split(','):
                        if fnmatch.fnmatch(fn, x):
                            found = True
                    if not found:
                        continue
                if exclude:
                    found = False
                    for x in exclude.split(','):
                        if fnmatch.fnmatch(fn, x):
                            found = True
                    if found:
                        continue
                chopped_path = os.path.relpath(fullpath, start=fileroot)
                fstat = os.stat(fullpath)
                f_size = fstat[osstat.ST_SIZE]
                f_modified_epoch = fstat[osstat.ST_MTIME]
                ret.append({'fullpath': fullpath, 'chopped_path': chopped_path, 'modified_epoch': f_modified_epoch, 'bytes': f_size})
    return ret