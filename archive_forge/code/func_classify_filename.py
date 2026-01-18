import os.path
from ... import urlutils
from ...trace import mutter
def classify_filename(name):
    """Classify a file based on its name.

    :param name: File path.
    :return: One of code, documentation, translation or art.
        None if determining the file type failed.
    """
    extension = os.path.splitext(name)[1]
    if extension in ('.c', '.h', '.py', '.cpp', '.rb', '.pm', '.pl', '.ac', '.java', '.cc', '.proto', '.yy', '.l'):
        return 'code'
    if extension in ('.html', '.xml', '.txt', '.rst', '.TODO'):
        return 'documentation'
    if extension in ('.po',):
        return 'translation'
    if extension in ('.svg', '.png', '.jpg'):
        return 'art'
    if not extension:
        basename = urlutils.basename(name)
        if basename in ('README', 'NEWS', 'TODO', 'AUTHORS', 'COPYING'):
            return 'documentation'
        if basename in ('Makefile',):
            return 'code'
    mutter("don't know how to classify %s", name)
    return None