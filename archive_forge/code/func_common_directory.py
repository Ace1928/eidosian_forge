def common_directory(paths):
    """Find the deepest common directory of a list of paths.

    :return: if no paths are provided, None is returned;
      if there is no common directory, '' is returned;
      otherwise the common directory with a trailing / is returned.
    """
    import posixpath

    def get_dir_with_slash(path):
        if path == b'' or path.endswith(b'/'):
            return path
        else:
            dirname, basename = posixpath.split(path)
            if dirname == b'':
                return dirname
            else:
                return dirname + b'/'
    if not paths:
        return None
    elif len(paths) == 1:
        return get_dir_with_slash(paths[0])
    else:
        common = common_path(paths[0], paths[1])
        for path in paths[2:]:
            common = common_path(common, path)
        return get_dir_with_slash(common)