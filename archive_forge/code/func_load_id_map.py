import os
def load_id_map(filename):
    """Load the mapping of commit ids to revision ids from a file.

    If the file does not exist, an empty result is returned.
    If the file does exists but cannot be opened, read or closed,
    the normal exceptions are thrown.

    NOTE: It is assumed that commit-ids do not have embedded spaces.

    :param filename: name of the file to save the data to
    :result: map, count where:
      map = a dictionary of commit ids to revision ids;
      count = the number of keys in map
    """
    result = {}
    count = 0
    if os.path.exists(filename):
        with open(filename) as f:
            for line in f:
                parts = line[:-1].split(' ', 1)
                result[parts[0]] = parts[1]
                count += 1
    return (result, count)