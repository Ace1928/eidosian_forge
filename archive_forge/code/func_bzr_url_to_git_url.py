from dulwich.client import parse_rsync_url
from .. import urlutils
from .refs import ref_to_branch_name
def bzr_url_to_git_url(location):
    target_url, target_params = urlutils.split_segment_parameters(location)
    branch = target_params.get('branch')
    ref = target_params.get('ref')
    return (target_url, branch, ref)