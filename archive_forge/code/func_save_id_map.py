import os
def save_id_map(filename, revision_ids):
    """Save the mapping of commit ids to revision ids to a file.

    Throws the usual exceptions if the file cannot be opened,
    written to or closed.

    :param filename: name of the file to save the data to
    :param revision_ids: a dictionary of commit ids to revision ids.
    """
    with open(filename, 'wb') as f:
        for commit_id in revision_ids:
            f.write(b'%s %s\n' % (commit_id, revision_ids[commit_id]))