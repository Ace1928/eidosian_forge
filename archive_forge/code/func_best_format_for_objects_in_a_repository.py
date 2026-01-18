import stat
from ... import controldir
def best_format_for_objects_in_a_repository(repo):
    """Find the high-level format for branches and trees given a repository.

    When creating branches and working trees within a repository, Bazaar
    defaults to using the default format which may not be the best choice.
    This routine does a reverse lookup of the high-level format registry
    to find the high-level format that a shared repository was most likely
    created via.

    :return: the BzrDirFormat or None if no matches were found.
    """
    repo_format = repo._format
    candidates = []
    non_aliases = set(controldir.format_registry.keys())
    non_aliases.difference_update(controldir.format_registry.aliases())
    for key in non_aliases:
        format = controldir.format_registry.make_controldir(key)
        if hasattr(format, 'repository_format'):
            if format.repository_format == repo_format:
                candidates.append((key, format))
    if len(candidates):
        name, format = candidates[0]
        return format
    else:
        return None