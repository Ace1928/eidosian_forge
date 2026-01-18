import urllib.parse
def format_hg_metadata(renames, branch, extra):
    """Construct a tail with hg-git metadata.

    :param renames: List of (oldpath, newpath) tuples with file renames
    :param branch: Branch name
    :param extra: Dictionary with extra data
    :return: Tail for commit message
    """
    extra_message = ''
    if branch != 'default':
        extra_message += 'branch : ' + branch + '\n'
    if renames:
        for oldfile, newfile in renames:
            extra_message += 'rename : ' + oldfile + ' => ' + newfile + '\n'
    for key, value in extra.iteritems():
        if key in ('author', 'committer', 'encoding', 'message', 'branch', 'hg-git'):
            continue
        else:
            extra_message += 'extra : ' + key + ' : ' + urllib.parse.quote(value) + '\n'
    if extra_message:
        return '\n--HG--\n' + extra_message
    else:
        return ''