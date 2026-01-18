from _pydev_bundle._pydev_filesystem_encoding import getfilesystemencoding
def constant_to_str(constant):
    s = ID_TO_MEANING.get(str(constant))
    if not s:
        s = '<Unknown: %s>' % (constant,)
    return s