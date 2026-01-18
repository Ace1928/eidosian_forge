def map_file_ids(repository, old_parents, new_parents):
    """Try to determine the equivalent file ids in two sets of parents.

    :param repository: Repository to use
    :param old_parents: List of revision ids of old parents
    :param new_parents: List of revision ids of new parents
    """
    assert len(old_parents) == len(new_parents)
    ret = {}
    for oldp, newp in zip(old_parents, new_parents):
        oldtree = repository.revision_tree(oldp)
        newtree = repository.revision_tree(newp)
        for path, ie in oldtree.iter_entries_by_dir():
            file_id = newtree.path2id(path)
            if file_id is not None:
                ret[ie.file_id] = file_id
    return ret