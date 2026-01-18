from rdkit.sping.colors import *
def getFileObject(file, openFlags='wb'):
    """Common code for every Canvas.save() operation takes a string
          or a potential file object and assures that a valid fileobj is returned"""
    if file:
        if isinstance(file, str):
            fileobj = open(file, openFlags)
        elif hasattr(file, 'write'):
            fileobj = file
        else:
            raise ValueError('Invalid file argument to save')
    else:
        raise ValueError('Invalid file argument to save')
    return fileobj