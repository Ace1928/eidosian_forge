from fontTools.misc.textTools import Tag, bytesjoin, strjoin
def setMacCreatorAndType(path, fileCreator, fileType):
    """Set file creator and file type codes for a path.

    Note that if the ``xattr`` module is not installed, no action is
    taken but no error is raised.

    Args:
            path (str): A file path.
            fileCreator: A four-character file creator tag.
            fileType: A four-character file type tag.

    """
    if xattr is not None:
        from fontTools.misc.textTools import pad
        if not all((len(s) == 4 for s in (fileCreator, fileType))):
            raise TypeError('arg must be string of 4 chars')
        finderInfo = pad(bytesjoin([fileType, fileCreator]), 32)
        xattr.setxattr(path, 'com.apple.FinderInfo', finderInfo)