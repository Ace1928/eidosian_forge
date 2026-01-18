from fontTools.misc.textTools import Tag, bytesjoin, strjoin
def getMacCreatorAndType(path):
    """Returns file creator and file type codes for a path.

    Args:
            path (str): A file path.

    Returns:
            A tuple of two :py:class:`fontTools.textTools.Tag` objects, the first
            representing the file creator and the second representing the
            file type.
    """
    if xattr is not None:
        try:
            finderInfo = xattr.getxattr(path, 'com.apple.FinderInfo')
        except (KeyError, IOError):
            pass
        else:
            fileType = Tag(finderInfo[:4])
            fileCreator = Tag(finderInfo[4:8])
            return (fileCreator, fileType)
    return (None, None)